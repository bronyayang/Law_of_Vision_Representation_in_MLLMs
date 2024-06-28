from diffusers import StableDiffusion3Pipeline
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
import torch
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.utils import is_torch_version
import gc
from transformers import CLIPTokenizer


class MySD3Transformer2DModell(SD3Transformer2DModel):
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        up_ft_indices,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        """
        The [`SD3Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        height, width = hidden_states.shape[-2:]

        hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        up_ft = {}
        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    **ckpt_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb
                )
            
            # handle negative indices
            if i-len(self.transformer_blocks) in up_ft_indices:
                up_ft[i-len(self.transformer_blocks)] = hidden_states.detach()
            elif i in up_ft_indices:
                up_ft[i] = hidden_states.detach()
            
        output = {}
        output['up_ft'] = up_ft
        return output



class OneStepDiTPipeline(StableDiffusion3Pipeline):
    @torch.no_grad()
    def __call__(
        self,
        img_tensor,
        t,
        prompt_embeds,
        pooled_prompt_embeds,
        up_ft_indices,
    ):
        
        device = self._execution_device
        # may affect performance
        # self.vae.to(dtype=img_tensor.dtype)
        latents = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor
        # img_tensor.to(torch.float, device=device)
        B = img_tensor.shape[0]
        t = torch.full((B,), t, dtype=torch.long, device=device)
        noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        # self.unet.to(dtype=latents_noisy.dtype)
        # latents_noisy.to(dtype=torch.float32)
        unet_output = self.transformer(latents_noisy, 
                                        pooled_projections=pooled_prompt_embeds, 
                                        encoder_hidden_states=prompt_embeds, 
                                        up_ft_indices=up_ft_indices, 
                                        timestep=t,
                                        joint_attention_kwargs=None)
        # unet_output.to(dtype=img_tensor.dtype)
        return unet_output


class SD3Featurizer:
    def __init__(self, sd_id="stabilityai/stable-diffusion-3-medium-diffusers"):
        # transformer = torch.compile(MySD3Transformer2DModell.from_pretrained(sd_id, subfolder="transformer", torch_dtype=torch.bfloat16), mode="max-autotune", fullgraph=True)
        transformer = MySD3Transformer2DModell.from_pretrained(sd_id, subfolder="transformer", torch_dtype=torch.bfloat16)
        # tokenizer = CLIPTokenizer.from_pretrained(sd_id, subfolder="tokenizer", torch_dtype=torch.bfloat16)
        onestep_pipe = OneStepDiTPipeline.from_pretrained(sd_id, 
                                                        transformer=transformer, 
                                                        text_encoder_3=None,
                                                        tokenizer_3=None,
                                                        torch_dtype=torch.bfloat16, 
                                                        low_cpu_mem_usage=False)
        onestep_pipe.vae.decoder = None
        # onestep_pipe.scheduler = DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler")
        gc.collect()
        onestep_pipe = onestep_pipe.to("cuda")
        onestep_pipe.enable_attention_slicing()
        # onestep_pipe.enable_xformers_memory_efficient_attention()
        self.pipe = onestep_pipe

    @torch.no_grad()
    def forward(self, img_tensor, prompt, t=1, up_ft_index=-1, ensemble_size=1):
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
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.pipe.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                prompt_3=None,
                device='cuda',
                num_images_per_prompt=1,
                do_classifier_free_guidance=False)
        unet_ft_all = self.pipe(
            img_tensor=img_tensor,
            t=t,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            up_ft_indices=[up_ft_index])
        unet_ft = unet_ft_all['up_ft'][up_ft_index]  # [B*ensem, c, h, w]
        h = w = int(unet_ft.shape[-2] ** 0.5)
        unet_ft = unet_ft.transpose(2,1).reshape(B, -1, h, w)
        unet_ft = unet_ft.unfold(3, 2, 2).unfold(2, 2, 2) # [B, c, h//2, w//2, 2, 2]
        unet_ft = unet_ft.reshape(B, -1, h//2, w//2, 4).permute(0, 4, 1, 2, 3).reshape(B, -1, h//2, w//2)
        # dit does not enable ensemble
        return unet_ft