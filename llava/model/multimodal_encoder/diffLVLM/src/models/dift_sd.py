from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import torch
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers import DDIMScheduler
import gc

class MyUNet2DConditionModel(UNet2DConditionModel):
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        up_ft_indices,
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None):
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            # logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        # 5. up
        up_ft = {}
        for i, upsample_block in enumerate(self.up_blocks):
            
            if i > np.max(up_ft_indices):
                break
            
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )
            
            if i in up_ft_indices:
                up_ft[i] = sample.detach()
            
        output = {}
        output['up_ft'] = up_ft
        return output

class OneStepSDPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        img_tensor,
        t,
        up_ft_indices,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None
    ):
        
        device = self._execution_device
        # may affect performance
        # self.vae.to(dtype=img_tensor.dtype)
        latents = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor
        # img_tensor.to(torch.float, device=device)
        t = torch.tensor(t, dtype=torch.long, device=device)
        noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        # self.unet.to(dtype=latents_noisy.dtype)
        # latents_noisy.to(dtype=torch.float32)
        unet_output = self.unet(latents_noisy, 
                            timestep=t,
                            up_ft_indices=up_ft_indices,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=cross_attention_kwargs)
        # unet_output.to(dtype=img_tensor.dtype)
        return unet_output


class OneStepSDXLPipeline(StableDiffusionXLPipeline):
    @torch.no_grad()
    def __call__(
        self,
        img_tensor,
        t,
        up_ft_indices,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None
    ):
        
        device = self._execution_device
        # may affect performance
        # self.vae.to(dtype=img_tensor.dtype)
        latents = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor
        # img_tensor.to(torch.float, device=device)
        t = torch.tensor(t, dtype=torch.long, device=device)
        noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        # self.unet.to(dtype=latents_noisy.dtype)
        # latents_noisy.to(dtype=torch.float32)
        unet_output = self.unet(latents_noisy, 
                            t,
                            up_ft_indices,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=cross_attention_kwargs)
        # unet_output.to(dtype=img_tensor.dtype)
        return unet_output

class SDFeaturizer:
    def __init__(self, sd_id='stabilityai/stable-diffusion-2-1'):
        unet = MyUNet2DConditionModel.from_pretrained(sd_id, subfolder="unet", torch_dtype=torch.bfloat16)
        if 'xl' in sd_id:
            onestep_pipe = OneStepSDXLPipeline.from_pretrained(sd_id, unet=unet, safety_checker=None, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False)
        else:
            onestep_pipe = OneStepSDPipeline.from_pretrained(sd_id, unet=unet, safety_checker=None, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False)
        onestep_pipe.vae.decoder = None
        onestep_pipe.scheduler = DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler")
        gc.collect()
        onestep_pipe = onestep_pipe.to("cuda")
        onestep_pipe.enable_attention_slicing()
        # onestep_pipe.enable_xformers_memory_efficient_attention()
        self.pipe = onestep_pipe

    @torch.no_grad()
    def forward(self, img_tensor, prompt, t=1, up_ft_index=0, ensemble_size=1):
        '''
        Args:
            img_tensor: should be a batch of torch tensors in the shape of [B, C, H, W]
            prompt: the prompt to use, a string
            t: the time step to use, should be an int in the range of [0, 1000]
            up_ft_index: which upsampling block of the U-Net to extract feature, you can choose [0, 1, 2, 3]
            ensemble_size: the number of repeated images used in the batch to extract features
        Return:
            unet_ft: a torch tensor in the shape of [B, 1, c, h, w]
        '''
        B, C, H, W = img_tensor.shape
        img_tensor = img_tensor.repeat_interleave(ensemble_size, dim=0).cuda()  # [B*ensem, C, H, W]
        if type(self.pipe) is OneStepSDXLPipeline:
            prompt_embeds = self.pipe.encode_prompt(
                prompt=prompt,
                device='cuda',
                num_images_per_prompt=1,
                do_classifier_free_guidance=False)[0]
        else:
            prompt_embeds = self.pipe.encode_prompt(
                prompt=prompt,
                device='cuda',
                num_images_per_prompt=1,
                do_classifier_free_guidance=False)  # [1, 77, dim]
        prompt_embeds = prompt_embeds[0].repeat(B * ensemble_size, 1, 1)
        unet_ft_all = self.pipe(
            img_tensor=img_tensor,
            t=t,
            up_ft_indices=[up_ft_index],
            prompt_embeds=prompt_embeds)
        unet_ft = unet_ft_all['up_ft'][up_ft_index]  # [B*ensem, c, h, w]
        _, _, h, w = unet_ft.shape

        # Reshape to separate batches and ensemble, then take the mean across the ensemble
        unet_ft = unet_ft.view(B, ensemble_size, -1, h, w).mean(1, keepdim=True)  # [B, 1, c, h, w]
        return unet_ft.squeeze()