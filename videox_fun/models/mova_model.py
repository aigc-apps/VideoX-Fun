# Modified from hhttps://github.com/OpenMOSS/MOVA/blob/main/mova/diffusion/pipelines/pipeline_mova.py
import math
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
    MOVA helper class that encapsulates transformer, transformer_2, transformer_audio, and dual_tower_bridge.
    Provides a clean forward interface similar to LTX2VideoTransformer3DModel.
    This is NOT an nn.Module, just a helper class for organizing forward logic.
    """
    def __init__(self, transformer, transformer_2, transformer_audio, dual_tower_bridge):
        super().__init__()
        self.transformer = transformer
        self.transformer_2 = transformer_2
        self.transformer_audio = transformer_audio
        self.dual_tower_bridge = dual_tower_bridge
        self.gradient_checkpointing = False
        self.model_offload = False  # Enable offloading unused models to CPU
    
    @property
    def dtype(self):
        """Return the dtype of the model (from first available transformer)."""
        if self.transformer is not None:
            return self.transformer.dtype
        elif self.transformer_2 is not None:
            return self.transformer_2.dtype
        else:
            raise AttributeError("MOVAModel has no available transformer to determine dtype")
    
    @property
    def config(self):
        """Return the config of the model (from first available transformer)."""
        if self.transformer is not None:
            return self.transformer.config
        elif self.transformer_2 is not None:
            return self.transformer_2.config
        else:
            raise AttributeError("MOVAModel has no available transformer to determine config")
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for all sub-models to save memory."""
        self.gradient_checkpointing = True
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False
    
    def enable_model_offload(self):
        """Enable model offloading to save VRAM.
        
        When enabled, only the active visual DiT (transformer or transformer_2) 
        and required components stay on GPU during forward pass.
        """
        self.model_offload = True
    
    def disable_model_offload(self):
        """Disable model offloading."""
        self.model_offload = False
    
    def set_module(self, module, module_name):
        """Disable model offloading."""
        setattr(self, module_name, module)
    
    def _move_to_device(self, model, device):
        """Helper to move model to device."""
        if model is not None and model.device != torch.device(type="meta"):
            model.to(device)
        return model
    
    def __call__(
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
            use_low_noise_dit: whether to use transformer (low noise, small t)
        
        Returns:
            visual_output: [B, C_visual, T_v, H_v, W_v]
            audio_output: [B, C_audio, T_a]
        """
        device = visual_latents.device
        
        # Select which visual DiT to use
        # Wan2.2 convention: transformer_2 = high-noise (large t), transformer = low-noise (small t)
        active_visual_dit = self.transformer if use_low_noise_dit else self.transformer_2
        inactive_visual_dit = self.transformer_2 if use_low_noise_dit else self.transformer
        
        # Check if active model is available
        if active_visual_dit is None:
            raise ValueError(
                f"Active visual DiT is None. use_low_noise_dit={use_low_noise_dit}. "
                f"This may happen when training with boundary_type='low' or 'high'. "
                f"Please check your training configuration."
            )
        
        # Model offloading: move inactive models to CPU to save VRAM
        if self.model_offload:
            # Move inactive visual DiT to CPU
            if inactive_visual_dit is not None:
                inactive_visual_dit.to('cpu')
            torch.cuda.empty_cache()
            
            # Move active visual DiT and transformer_audio to GPU
            active_visual_dit = self._move_to_device(active_visual_dit, device)
            self.transformer_audio = self._move_to_device(self.transformer_audio, device)
            self.dual_tower_bridge = self._move_to_device(self.dual_tower_bridge, device)
        else:
            # No offload: just ensure models are on correct device
            active_visual_dit = self._move_to_device(active_visual_dit, device)
            self.transformer_audio = self._move_to_device(self.transformer_audio, device)
            self.dual_tower_bridge = self._move_to_device(self.dual_tower_bridge, device)
        
        output = self._forward_single_step(
            visual_dit=active_visual_dit,
            visual_latents=visual_latents,
            audio_latents=audio_latents,
            context=context,
            timestep=timestep,
            audio_timestep=audio_timestep,
            frame_rate=frame_rate,
        )
        
        # Move active models back to CPU if offloading is enabled
        if self.model_offload:
            active_visual_dit.to('cpu')
            self.transformer_audio.to('cpu')
            self.dual_tower_bridge.to('cpu')
            torch.cuda.empty_cache()
        
        return output
    
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

            audio_t = self.transformer_audio.time_embedding(sinusoidal_embedding_1d(self.transformer_audio.freq_dim, audio_timestep))
            audio_t_mod = self.transformer_audio.time_projection(audio_t).unflatten(1, (6, self.transformer_audio.dim))
        
        model_dtype = visual_dit.dtype
        visual_t = visual_t.to(model_dtype)
        visual_t_mod = visual_t_mod.to(model_dtype)
        audio_t = audio_t.to(model_dtype)
        audio_t_mod = audio_t_mod.to(model_dtype)
        
        # Context embeddings
        visual_context_emb = visual_dit.text_embedding(visual_context)
        audio_context_emb = self.transformer_audio.text_embedding(audio_context)
        
        visual_x = visual_latents.to(model_dtype)
        audio_x = audio_latents.to(model_dtype)

        # Visual patchify
        visual_x = visual_x.contiguous(memory_format=torch.channels_last_3d)
        visual_x = visual_dit.patch_embedding(visual_x)
        grid_size = visual_x.shape[2:]
        visual_x = rearrange(visual_x, 'b c f h w -> b (f h w) c').contiguous()
        t, h, w = grid_size

        # Audio patchify
        audio_x = self.transformer_audio.patch_embedding(audio_x)
        audio_grid_size = audio_x.shape[2:]
        audio_x = rearrange(audio_x, 'b c f -> b f c').contiguous()
        f = audio_grid_size[0]

        # Audio freqs
        audio_freqs = torch.cat(
            [
                self.transformer_audio.freqs[0][:f].view(f, -1).expand(f, -1),
                self.transformer_audio.freqs[1][:f].view(f, -1).expand(f, -1),
                self.transformer_audio.freqs[2][:f].view(f, -1).expand(f, -1),
            ],
            dim=-1
        ).reshape(f, 1, -1).to(audio_x.device)

        # Sequence parallel: chunk visual_x before blocks
        # sp_world_size and sp_world_rank are also used in _forward_dual_tower_dit
        self._sp_world_size = getattr(visual_dit, 'sp_world_size', 1)
        self._sp_world_rank = getattr(visual_dit, 'sp_world_rank', 0)
        if self._sp_world_size > 1:
            # Pad sequence to be divisible by sp_world_size
            seq_len = visual_x.shape[1]
            padded_seq_len = int(math.ceil(seq_len / self._sp_world_size)) * self._sp_world_size
            if padded_seq_len > seq_len:
                visual_x = torch.cat([
                    visual_x,
                    visual_x.new_zeros(visual_x.shape[0], padded_seq_len - seq_len, visual_x.shape[2])
                ], dim=1)
            # Chunk for sequence parallel
            visual_x = torch.chunk(visual_x, self._sp_world_size, dim=1)[self._sp_world_rank]
        
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
        
        # Sequence parallel: all_gather visual output after blocks
        if self._sp_world_size > 1 and hasattr(visual_dit, 'all_gather') and visual_dit.all_gather is not None:
            visual_x = visual_dit.all_gather(visual_x, dim=1)
        
        # Visual head + unpatchify
        visual_output = visual_dit.head(visual_x, visual_t)
        grid_sizes_tensor = torch.tensor([grid_size], dtype=torch.long, device=visual_output.device)
        visual_output = visual_dit.unpatchify(visual_output, grid_sizes_tensor)
        visual_output = visual_output[0].unsqueeze(0)
        
        # Audio head + unpatchify
        audio_output = self.transformer_audio.head(audio_x, audio_t)
        audio_output = self.transformer_audio.unpatchify(audio_output, (f, ))

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
        min_layers = min(len(visual_dit.blocks), len(self.transformer_audio.blocks))
        visual_layers = len(visual_dit.blocks)

        # Check if sequence parallel is enabled
        sp_world_size = getattr(visual_dit, 'sp_world_size', 1)
        sp_world_rank = getattr(visual_dit, 'sp_world_rank', 0)
        sp_enabled = sp_world_size > 1 and hasattr(visual_dit, 'all_gather') and visual_dit.all_gather is not None

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
            self.transformer_audio.freqs[0][:audio_f].view(audio_f, -1),
            self.transformer_audio.freqs[1][:audio_f].view(audio_f, -1),
            self.transformer_audio.freqs[2][:audio_f].view(audio_f, -1),
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
            audio_block = self.transformer_audio.blocks[layer_idx]

            # Cross-modal interaction via bridge with optional gradient checkpointing
            # For sequence parallel: v2a (visual->audio) needs full visual sequence as key/value
            # So we all_gather visual_x before bridge, then chunk it back after
            needs_interaction = (
                self.dual_tower_bridge.should_interact(layer_idx, 'a2v') or 
                self.dual_tower_bridge.should_interact(layer_idx, 'v2a')
            )
            
            if needs_interaction:
                # Prepare visual_x for bridge: all_gather if sequence parallel is enabled
                if sp_enabled:
                    visual_x_for_bridge = visual_dit.all_gather(visual_x, dim=1)
                else:
                    visual_x_for_bridge = visual_x
                
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    def create_custom_forward(module):
                        def custom_forward(layer_idx, visual_x, audio_x, x_freqs, y_freqs, 
                                         a2v_condition_scale, v2a_condition_scale, 
                                         condition_scale, video_grid_size):
                            return module(
                                layer_idx,
                                visual_x,
                                audio_x,
                                x_freqs=x_freqs,
                                y_freqs=y_freqs,
                                a2v_condition_scale=a2v_condition_scale,
                                v2a_condition_scale=v2a_condition_scale,
                                condition_scale=condition_scale,
                                video_grid_size=video_grid_size,
                            )
                        return custom_forward
                    
                    visual_x_out, audio_x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.dual_tower_bridge),
                        layer_idx,
                        visual_x_for_bridge,
                        audio_x,
                        visual_rope_cos_sin,
                        audio_rope_cos_sin,
                        a2v_condition_scale,
                        v2a_condition_scale,
                        condition_scale,
                        grid_size,
                        use_reentrant=False,
                    )
                else:
                    visual_x_out, audio_x = self.dual_tower_bridge(
                        layer_idx,
                        visual_x_for_bridge,
                        audio_x,
                        x_freqs=visual_rope_cos_sin,
                        y_freqs=audio_rope_cos_sin,
                        a2v_condition_scale=a2v_condition_scale,
                        v2a_condition_scale=v2a_condition_scale,
                        condition_scale=condition_scale,
                        video_grid_size=grid_size,
                    )
                
                # Chunk visual_x back to local rank if sequence parallel is enabled
                # Bridge output visual_x might be modified (a2v direction), so always chunk
                if sp_enabled:
                    visual_x = torch.chunk(visual_x_out, sp_world_size, dim=1)[sp_world_rank]
                else:
                    visual_x = visual_x_out

            # Visual block with optional gradient checkpointing
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(visual_x, visual_t_mod, visual_seq_lens, visual_grid_sizes,
                                     wan_freqs, visual_context, visual_context_lens, visual_dtype):
                        return module(
                            visual_x,
                            e=visual_t_mod,
                            seq_lens=visual_seq_lens,
                            grid_sizes=visual_grid_sizes,
                            freqs=wan_freqs,
                            context=visual_context,
                            context_lens=visual_context_lens,
                            dtype=visual_dtype,
                        )
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
                    def custom_forward(audio_x, audio_t_mod, audio_seq_lens, audio_grid_sizes,
                                     audio_freqs_dit, audio_context, audio_context_lens, audio_dtype):
                        return module(
                            audio_x,
                            e=audio_t_mod,
                            seq_lens=audio_seq_lens,
                            grid_sizes=audio_grid_sizes,
                            freqs=audio_freqs_dit,
                            context=audio_context,
                            context_lens=audio_context_lens,
                            dtype=audio_dtype,
                        )
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
                    def custom_forward(visual_x, visual_t_mod, visual_seq_lens, visual_grid_sizes,
                                     wan_freqs, visual_context, visual_context_lens, visual_dtype):
                        return module(
                            visual_x,
                            e=visual_t_mod,
                            seq_lens=visual_seq_lens,
                            grid_sizes=visual_grid_sizes,
                            freqs=wan_freqs,
                            context=visual_context,
                            context_lens=visual_context_lens,
                            dtype=visual_dtype,
                        )
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
