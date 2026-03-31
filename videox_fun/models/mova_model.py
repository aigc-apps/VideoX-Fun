# Modified from hhttps://github.com/OpenMOSS/MOVA/blob/main/mova/diffusion/pipelines/pipeline_mova.py
import torch
import torch.nn as nn
from einops import rearrange

def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


class MOVAModel(nn.Module):
    """
    MOVA model that encapsulates transformer, transformer_2, audio_dit, and dual_tower_bridge.
    Provides a clean forward interface similar to LTX2VideoTransformer3DModel.
    """
    def __init__(self, transformer, transformer_2, audio_dit, dual_tower_bridge):
        super().__init__()
        self.transformer = transformer
        self.transformer_2 = transformer_2
        self.audio_dit = audio_dit
        self.dual_tower_bridge = dual_tower_bridge
        self.gradient_checkpointing = False
    
    @property
    def dtype(self):
        """Return the dtype of the model (from transformer)."""
        return self.transformer.dtype
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for all sub-models to save memory."""
        self.gradient_checkpointing = True
    
    def forward(
        self,
        visual_latents: torch.Tensor,
        audio_latents: torch.Tensor,
        context: torch.Tensor,
        timestep: torch.Tensor,
        audio_timestep: torch.Tensor,
        frame_rate: float,
        use_low_noise_dit: bool = False,
    ):
        """
        Forward pass for MOVA model.
        
        Args:
            visual_latents: [B, C_visual, T_v, H_v, W_v]
            audio_latents: [B, C_audio, T_a]
            context: [B, L_context, C_context]
            timestep: [B] or scalar
            audio_timestep: [B] or scalar
            frame_rate: float
            use_low_noise_dit: whether to use transformer_2 (low noise)
        
        Returns:
            visual_output: [B, C_visual, T_v, H_v, W_v]
            audio_output: [B, C_audio, T_a]
        """
        # Select which visual DiT to use
        visual_dit = self.transformer_2 if use_low_noise_dit else self.transformer
        
        return self._forward_single_step(
            visual_dit=visual_dit,
            visual_latents=visual_latents,
            audio_latents=audio_latents,
            context=context,
            timestep=timestep,
            audio_timestep=audio_timestep,
            frame_rate=frame_rate,
        )
    
    def _forward_single_step(
        self,
        visual_dit,
        visual_latents: torch.Tensor,
        audio_latents: torch.Tensor,
        context: torch.Tensor,
        timestep: torch.Tensor,
        audio_timestep: torch.Tensor,
        frame_rate: float,
    ):
        """Single step forward pass."""
        visual_x = visual_latents
        audio_x = audio_latents
        audio_context = visual_context = context

        if audio_timestep is None:
            audio_timestep = timestep

        # Time embeddings
        with torch.autocast("cuda", dtype=torch.float32):
            visual_t = visual_dit.time_embedding(sinusoidal_embedding_1d(visual_dit.freq_dim, timestep))
            visual_t_mod = visual_dit.time_projection(visual_t).unflatten(1, (6, visual_dit.dim))

            audio_t = self.audio_dit.time_embedding(sinusoidal_embedding_1d(self.audio_dit.freq_dim, audio_timestep))
            audio_t_mod = self.audio_dit.time_projection(audio_t).unflatten(1, (6, self.audio_dit.dim))
        
        model_dtype = visual_dit.dtype
        visual_t = visual_t.to(model_dtype)
        visual_t_mod = visual_t_mod.to(model_dtype)
        audio_t = audio_t.to(model_dtype)
        audio_t_mod = audio_t_mod.to(model_dtype)
        
        # Context embeddings
        visual_context_emb = visual_dit.text_embedding(visual_context)
        audio_context_emb = self.audio_dit.text_embedding(audio_context)
        
        visual_x = visual_latents.to(model_dtype)
        audio_x = audio_latents.to(model_dtype)

        # Visual patchify
        visual_x = visual_x.contiguous(memory_format=torch.channels_last_3d)
        visual_x = visual_dit.patch_embedding(visual_x)
        grid_size = visual_x.shape[2:]
        visual_x = rearrange(visual_x, 'b c f h w -> b (f h w) c').contiguous()
        t, h, w = grid_size

        # Audio patchify
        audio_x = self.audio_dit.patch_embedding(audio_x)
        audio_grid_size = audio_x.shape[2:]
        audio_x = rearrange(audio_x, 'b c f -> b f c').contiguous()
        f = audio_grid_size[0]

        # Audio freqs
        audio_freqs = torch.cat(
            [
                self.audio_dit.freqs[0][:f].view(f, -1).expand(f, -1),
                self.audio_dit.freqs[1][:f].view(f, -1).expand(f, -1),
                self.audio_dit.freqs[2][:f].view(f, -1).expand(f, -1),
            ],
            dim=-1
        ).reshape(f, 1, -1).to(audio_x.device)

        # Forward through dual tower DiT blocks
        visual_x, audio_x = self._forward_dual_tower_dit(
            visual_dit=visual_dit,
            visual_x=visual_x,
            audio_x=audio_x,
            visual_context=visual_context_emb,
            audio_context=audio_context_emb,
            visual_t_mod=visual_t_mod,
            audio_t_mod=audio_t_mod,
            grid_size=grid_size,
            frame_rate=frame_rate,
        )
        
        # Visual head + unpatchify
        visual_output = visual_dit.head(visual_x, visual_t)
        grid_sizes_tensor = torch.tensor([grid_size], dtype=torch.long, device=visual_output.device)
        visual_output = visual_dit.unpatchify(visual_output, grid_sizes_tensor)
        visual_output = visual_output[0].unsqueeze(0)
        
        # Audio head + unpatchify
        audio_output = self.audio_dit.head(audio_x, audio_t)
        audio_output = self.audio_dit.unpatchify(audio_output, (f, ))

        return visual_output, audio_output
    
    def _forward_dual_tower_dit(
        self,
        visual_dit,
        visual_x: torch.Tensor,
        audio_x: torch.Tensor,
        visual_context: torch.Tensor,
        audio_context: torch.Tensor,
        visual_t_mod: torch.Tensor,
        audio_t_mod: torch.Tensor,
        grid_size: tuple[int, int, int],
        frame_rate: float,
        condition_scale: float = 1.0,
        a2v_condition_scale: float = None,
        v2a_condition_scale: float = None,
    ):
        """Forward through dual tower DiT blocks with bridge."""
        min_layers = min(len(visual_dit.blocks), len(self.audio_dit.blocks))
        visual_layers = len(visual_dit.blocks)

        # Prepare visual block parameters
        t, h, w = grid_size
        seq_len = t * h * w
        visual_seq_lens = torch.tensor([seq_len], dtype=torch.long, device=visual_x.device)
        visual_grid_sizes = torch.tensor([[t, h, w]], dtype=torch.long, device=visual_x.device)
        visual_context_lens = None
        visual_dtype = visual_x.dtype
        wan_freqs = visual_dit.freqs.to(visual_x.device)

        # Prepare audio block parameters
        audio_f = audio_x.shape[1]
        audio_seq_lens = torch.tensor([audio_f], dtype=torch.long, device=audio_x.device)
        audio_grid_sizes = torch.tensor([[audio_f]], dtype=torch.long, device=audio_x.device)
        audio_context_lens = None
        audio_dtype = audio_x.dtype
        audio_freqs_dit = torch.cat([
            self.audio_dit.freqs[0][:audio_f].view(audio_f, -1),
            self.audio_dit.freqs[1][:audio_f].view(audio_f, -1),
            self.audio_dit.freqs[2][:audio_f].view(audio_f, -1),
        ], dim=-1).reshape(audio_f, 1, -1).to(audio_x.device)

        # Precompute cross-modal RoPE freqs
        if self.dual_tower_bridge.apply_cross_rope:
            (visual_rope_cos_sin, audio_rope_cos_sin) = self.dual_tower_bridge.build_aligned_freqs(
                frame_rate=frame_rate,
                grid_size=grid_size,
                audio_steps=audio_x.shape[1],
                device=visual_x.device,
                dtype=visual_x.dtype,
            )
        else:
            visual_rope_cos_sin = None
            audio_rope_cos_sin = None

        # Forward through blocks
        for layer_idx in range(min_layers):
            visual_block = visual_dit.blocks[layer_idx]
            audio_block = self.audio_dit.blocks[layer_idx]

            # Cross-modal interaction via bridge with optional gradient checkpointing
            if self.dual_tower_bridge.should_interact(layer_idx, 'a2v'):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)
                        return custom_forward
                    
                    visual_x, audio_x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.dual_tower_bridge),
                        layer_idx,
                        visual_x,
                        audio_x,
                        x_freqs=visual_rope_cos_sin,
                        y_freqs=audio_rope_cos_sin,
                        a2v_condition_scale=a2v_condition_scale,
                        v2a_condition_scale=v2a_condition_scale,
                        condition_scale=condition_scale,
                        video_grid_size=grid_size,
                        use_reentrant=False,
                    )
                else:
                    visual_x, audio_x = self.dual_tower_bridge(
                        layer_idx,
                        visual_x,
                        audio_x,
                        x_freqs=visual_rope_cos_sin,
                        y_freqs=audio_rope_cos_sin,
                        a2v_condition_scale=a2v_condition_scale,
                        v2a_condition_scale=v2a_condition_scale,
                        condition_scale=condition_scale,
                        video_grid_size=grid_size,
                    )

            # Visual block with optional gradient checkpointing
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                visual_x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(visual_block),
                    visual_x,
                    visual_t_mod,
                    visual_seq_lens,
                    visual_grid_sizes,
                    wan_freqs,
                    visual_context,
                    visual_context_lens,
                    visual_dtype,
                    use_reentrant=False,
                )
            else:
                visual_x = visual_block(
                    visual_x,
                    e=visual_t_mod,
                    seq_lens=visual_seq_lens,
                    grid_sizes=visual_grid_sizes,
                    freqs=wan_freqs,
                    context=visual_context,
                    context_lens=visual_context_lens,
                    dtype=visual_dtype,
                )
            
            # Audio block with optional gradient checkpointing
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                audio_x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(audio_block),
                    audio_x,
                    audio_t_mod,
                    audio_seq_lens,
                    audio_grid_sizes,
                    audio_freqs_dit,
                    audio_context,
                    audio_context_lens,
                    audio_dtype,
                    use_reentrant=False,
                )
            else:
                audio_x = audio_block(
                    audio_x,
                    e=audio_t_mod,
                    seq_lens=audio_seq_lens,
                    grid_sizes=audio_grid_sizes,
                    freqs=audio_freqs_dit,
                    context=audio_context,
                    context_lens=audio_context_lens,
                    dtype=audio_dtype,
                )
        
        # Forward remaining visual blocks
        for layer_idx in range(min_layers, visual_layers):
            visual_block = visual_dit.blocks[layer_idx]
            
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                visual_x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(visual_block),
                    visual_x,
                    visual_t_mod,
                    visual_seq_lens,
                    visual_grid_sizes,
                    wan_freqs,
                    visual_context,
                    visual_context_lens,
                    visual_dtype,
                    use_reentrant=False,
                )
            else:
                visual_x = visual_block(
                    visual_x,
                    e=visual_t_mod,
                    seq_lens=visual_seq_lens,
                    grid_sizes=visual_grid_sizes,
                    freqs=wan_freqs,
                    context=visual_context,
                    context_lens=visual_context_lens,
                    dtype=visual_dtype,
                )
        
        return visual_x, audio_x
