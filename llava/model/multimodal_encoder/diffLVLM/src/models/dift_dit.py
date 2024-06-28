from diffusers import DiTPipeline
import torch
from typing import Any, Dict, Optional
from diffusers.models.transformers.dit_transformer_2d import DiTTransformer2DModel
from diffusers.models.normalization import CombinedTimestepLabelEmbeddings
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version
import gc

class MyCombinedTimestepLabelEmbeddings(CombinedTimestepLabelEmbeddings):
    def forward(self, timestep, class_labels, hidden_dtype=None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)

        conditioning = timesteps_emb  # (N, D)

        return conditioning

class MyDiTTransformer2DModel(DiTTransformer2DModel):
    def forward(
        self,
        hidden_states: torch.Tensor,
        up_ft_indices,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None
    ):
        """
        The [`Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input `hidden_states`.
            encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        # 1. Input
        height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
        hidden_states = self.pos_embed(hidden_states)

        # 2. Blocks
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
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    None,
                    None,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                )
            
            # handle negative indices
            if i-len(self.transformer_blocks) in up_ft_indices:
                up_ft[i-len(self.transformer_blocks)] = hidden_states.detach()
            elif i in up_ft_indices:
                up_ft[i] = hidden_states.detach()
            
        output = {}
        output['up_ft'] = up_ft
        return output

class OneStepDiTPipeline(DiTPipeline):
    @torch.no_grad()
    def __call__(
        self,
        img_tensor,
        t,
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
        unet_output = self.transformer(latents_noisy, up_ft_indices=up_ft_indices, timestep=t)
        # unet_output.to(dtype=img_tensor.dtype)
        return unet_output


def replace_combined_timestep_label_embeddings(module):
    for name, child in module.named_children():
        if isinstance(child, CombinedTimestepLabelEmbeddings):
            # Extract the original num_classes and embedding_dim
            num_classes = child.class_embedder.num_classes
            embedding_dim = child.class_embedder.embedding_table.embedding_dim
            
            # Replace the class while keeping the weights
            new_emb = MyCombinedTimestepLabelEmbeddings(num_classes, embedding_dim)
            new_emb.load_state_dict(child.state_dict())
            setattr(module, name, new_emb)
        else:
            replace_combined_timestep_label_embeddings(child)

class DiTFeaturizer:
    def __init__(self, sd_id='facebook/DiT-XL-2-512'):
        transformer = MyDiTTransformer2DModel.from_pretrained(sd_id, subfolder="transformer", torch_dtype=torch.bfloat16)
        replace_combined_timestep_label_embeddings(transformer)
        transformer = transformer.to(torch.bfloat16)
        onestep_pipe = OneStepDiTPipeline.from_pretrained(sd_id, transformer=transformer, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False)
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
        unet_ft_all = self.pipe(
            img_tensor=img_tensor,
            t=t,
            up_ft_indices=[up_ft_index])
        unet_ft = unet_ft_all['up_ft'][up_ft_index]  # [B*ensem, c, h, w]
        h = w = int(unet_ft.shape[-2] ** 0.5)
        unet_ft = unet_ft.transpose(2,1).reshape(B, -1, h, w)
        unet_ft = unet_ft.unfold(3, 2, 2).unfold(2, 2, 2) # [B, c, h//2, w//2, 2, 2]
        unet_ft = unet_ft.reshape(B, -1, h//2, w//2, 4).permute(0, 4, 1, 2, 3).reshape(B, -1, h//2, w//2)
        # dit does not enable ensemble
        return unet_ft