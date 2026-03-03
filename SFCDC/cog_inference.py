"""
CogVideoX Text-to-Video generation with SFCDC sparse attention.

Uses the diffusers CogVideoXPipeline directly (no manual denoising loop).
SFCDC is integrated via a custom attention processor that handles CFG
batch splitting internally, keeping cond/uncond centroids independent.

Usage:
    python cog_t2v_inference.py --prompt "A girl riding a bike." --model_path THUDM/CogVideoX1.5-5B
"""

import argparse
import logging
import os
import sys
from typing import Optional

import torch
import torch.nn.functional as F

# === Path setup ===
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from SFCDC.sfcdc_v5 import SFCDC_Simulator

from diffusers import CogVideoXDPMScheduler, CogVideoXPipeline
from diffusers.utils import export_to_video
from diffusers.models.attention_processor import Attention

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Recommended resolution (height, width) for each model
RESOLUTION_MAP = {
    "cogvideox1.5-5b": (768, 1360),
    "cogvideox-5b": (480, 720),
    "cogvideox-2b": (480, 720),
}


# ===========================================================================
#  SFCDC Global State
# ===========================================================================
class SFCDCState:
    """Tracks the current denoising step index for SFCDC processors."""
    iter_idx: int = 0

sfcdc_state = SFCDCState()


# ===========================================================================
#  Custom Attention Processor with SFCDC (CFG-aware batch splitting)
# ===========================================================================
class CogVideoX_SFCDC_AttnProcessor:
    """
    Drop-in replacement for CogVideoXAttnProcessor2_0 that integrates SFCDC.

    When the pipeline uses CFG (batch_size == 2), this processor automatically
    splits the batch into uncond (batch[0]) and cond (batch[1]), processes each
    through SFCDC with independent clustering state, then recombines.

    When SFCDC is inactive: falls back to standard scaled_dot_product_attention.
    """

    def __init__(self, layer_idx: int, sfcdc_simulator: SFCDC_Simulator):
        self.layer_idx = layer_idx
        self.sfcdc_simulator = sfcdc_simulator
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoX_SFCDC_AttnProcessor requires PyTorch 2.0+")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        # Keep SFCDC token metadata aligned with runtime tensors.
        self.sfcdc_simulator.text_token_length = int(text_seq_length)

        # Concatenate text + video tokens (same as original CogVideoX)
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        batch_size, sequence_length, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        # QKV projection
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # Reshape: [B, L, H*D] -> [B, H, L, D]
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE (only to video part, same as original)
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query[:, :, text_seq_length:] = apply_rotary_emb(
                query[:, :, text_seq_length:], image_rotary_emb
            )
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(
                    key[:, :, text_seq_length:], image_rotary_emb
                )

        # ========== SFCDC Branch (CFG-aware) ==========
        if batch_size == 2:
            # CFG mode: batch[0]=uncond, batch[1]=cond
            # Process separately to maintain independent SFCDC clustering state
            cfg_keys = ["uncond", "cond"]
            results = []
            for b in range(2):
                q_b = query[b : b + 1].transpose(1, 2)   # [1, L, N, D]
                k_b = key[b : b + 1].transpose(1, 2)
                v_b = value[b : b + 1].transpose(1, 2)

                video_tokens_b = q_b.shape[1] - text_seq_length
                spatial_tokens = self.sfcdc_simulator.l_h * self.sfcdc_simulator.l_w
                if spatial_tokens > 0 and video_tokens_b > 0 and video_tokens_b % spatial_tokens == 0:
                    self.sfcdc_simulator.l_f = video_tokens_b // spatial_tokens

                sfcdc_out = self.sfcdc_simulator.analyze(
                    q=q_b, k=k_b, v=v_b,
                    layer_idx=self.layer_idx,
                    iter_idx=sfcdc_state.iter_idx,
                    key=cfg_keys[b],
                )
                if sfcdc_out is not None:
                    results.append(sfcdc_out.transpose(1, 2))  # [1, N, L, D]
                else:
                    mask_b = attention_mask[b : b + 1] if attention_mask is not None else None
                    results.append(F.scaled_dot_product_attention(
                        query[b : b + 1], key[b : b + 1], value[b : b + 1],
                        attn_mask=mask_b, dropout_p=0.0, is_causal=False,
                    ))
            hidden_states = torch.cat(results, dim=0)  # [2, N, L, D]
        else:
            # Non-CFG or single batch
            q_sfcdc = query.transpose(1, 2)   # [B, L, N, D]
            k_sfcdc = key.transpose(1, 2)
            v_sfcdc = value.transpose(1, 2)

            video_tokens = q_sfcdc.shape[1] - text_seq_length
            spatial_tokens = self.sfcdc_simulator.l_h * self.sfcdc_simulator.l_w
            if spatial_tokens > 0 and video_tokens > 0 and video_tokens % spatial_tokens == 0:
                self.sfcdc_simulator.l_f = video_tokens // spatial_tokens

            sfcdc_out = self.sfcdc_simulator.analyze(
                q=q_sfcdc, k=k_sfcdc, v=v_sfcdc,
                layer_idx=self.layer_idx,
                iter_idx=sfcdc_state.iter_idx,
                key="cond",
            )
            if sfcdc_out is not None:
                hidden_states = sfcdc_out.transpose(1, 2)  # [B, N, L, D]
            else:
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value,
                    attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
                )

        # Reshape back: [B, N, L, D] -> [B, L, N*D]
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )

        # Output projection + dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        # Split text and video outputs
        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


# ===========================================================================
#  Replace Attention Processors
# ===========================================================================
def replace_attention_processors(pipe, sfcdc_simulator: SFCDC_Simulator):
    """Replace all attention processors in CogVideoX transformer with SFCDC ones."""
    transformer = pipe.transformer
    num_layers = len(transformer.transformer_blocks)
    for layer_idx, block in enumerate(transformer.transformer_blocks):
        processor = CogVideoX_SFCDC_AttnProcessor(
            layer_idx=layer_idx,
            sfcdc_simulator=sfcdc_simulator,
        )
        block.attn1.set_processor(processor)
    logger.info(f"Replaced {num_layers} attention processors with SFCDC.")


# ===========================================================================
#  Compute CogVideoX Latent Geometry for SFCDC
# ===========================================================================
def compute_latent_geometry(pipe, height, width, num_frames):
    """
    Compute the latent video token geometry (l_f, l_h, l_w) for SFCDC.
    These correspond to the number of tokens along frame, height, and width
    dimensions in the transformer's latent space.
    """
    vae_scale_spatial = pipe.vae_scale_factor_spatial
    vae_scale_temporal = pipe.vae_scale_factor_temporal
    p = pipe.transformer.config.patch_size
    p_t = pipe.transformer.config.patch_size_t

    latent_h = height // vae_scale_spatial
    latent_w = width // vae_scale_spatial
    latent_frames = (num_frames - 1) // vae_scale_temporal + 1

    grid_h = latent_h // p
    grid_w = latent_w // p

    if p_t is not None:
        additional_frames = 0
        if latent_frames % p_t != 0:
            additional_frames = p_t - latent_frames % p_t
        grid_f = (latent_frames + additional_frames) // p_t
    else:
        grid_f = latent_frames

    text_token_length = 226
    logger.info(
        f"Latent geometry: l_f={grid_f}, l_h={grid_h}, l_w={grid_w}, "
        f"video_tokens={grid_f * grid_h * grid_w}, text_tokens={text_token_length}"
    )
    return grid_f, grid_h, grid_w, text_token_length


# ===========================================================================
#  Callback: update SFCDC state after each denoising step
# ===========================================================================
def _make_sfcdc_step_callback(sfcdc_simulator, num_inference_steps):
    """Create a pipeline callback that advances SFCDC iter_idx and clears cache after the last step."""
    def _sfcdc_step_callback(pipe, step_index, timestep, callback_kwargs):
        sfcdc_state.iter_idx = step_index + 1
        # After the last denoising step, clear SFCDC cache to free memory for VAE decoding
        if step_index + 1 >= num_inference_steps:
            sfcdc_simulator.clear_cache()
            logger.info("SFCDC cache cleared after final denoising step.")
        return callback_kwargs
    return _sfcdc_step_callback


# ===========================================================================
#  Main Generation Function
# ===========================================================================
def generate_video(
    prompt: str,
    model_path: str,
    output_path: str = "./output.mp4",
    num_frames: int = 81,
    width: Optional[int] = None,
    height: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    fps: int = 16,
    use_dynamic_cfg: bool = True,
    # SFCDC parameters
    sfcdc_enabled: bool = False,
    sfcdc_start_iter: int = 12,
    sfcdc_end_iter: int = 46,
    sfcdc_centers_q: int = 256,
    sfcdc_centers_k: int = 256,
    sfcdc_group_size: int = 4,
    sfcdc_k0: float = 0.25,
    sfcdc_k1: float = 0.10,
    sfcdc_k2: float = 0.10,
    # Prompt ID system
    prompt_path: str = "./tests/CogVideoX/prompts",
    prompt_id: Optional[int] = None,
):
    """Generate video using CogVideoX with SFCDC sparse attention."""

    # --- Handle prompt_id ---
    if prompt_id is not None:
        prompt_file = os.path.join(prompt_path, str(prompt_id), "prompt.txt")
        if os.path.isfile(prompt_file):
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            logger.info(f"Loaded prompt from {prompt_file}")
        else:
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    # --- Resolve resolution ---
    model_name = model_path.split("/")[-1].lower()
    default_h, default_w = RESOLUTION_MAP.get(model_name, (768, 1360))
    height = height or default_h
    width = width or default_w
    logger.info(f"Resolution: {width}x{height}, frames: {num_frames}")

    # --- Load Pipeline (same as cli_demo.py) ---
    logger.info(f"Loading CogVideoX pipeline from {model_path}...")
    pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    pipe.enable_model_cpu_offload()
    # pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    # pipe.vae.enable_tiling()

    # --- Setup SFCDC ---
    l_f, l_h, l_w, text_token_length = compute_latent_geometry(
        pipe, height, width, num_frames
    )
    sfcdc_simulator = SFCDC_Simulator(
        enabled=sfcdc_enabled,
        clusbegin_iter=sfcdc_start_iter - 1 if sfcdc_start_iter > 0 else 0,
        start_iter=sfcdc_start_iter,
        end_iter=sfcdc_end_iter,
        centers_q=sfcdc_centers_q,
        centers_k=sfcdc_centers_k,
        group_size=sfcdc_group_size,
        k0=sfcdc_k0, k1=sfcdc_k1, k2=sfcdc_k2,
        l_f=l_f, l_h=l_h, l_w=l_w,
        text_token_length=text_token_length,
    )
    logger.info(f"SFCDC Simulator: enabled={sfcdc_enabled}, text_tokens={text_token_length}")
    logger.info(f"SFCDC Parameters: start_iter={sfcdc_start_iter}, end_iter={sfcdc_end_iter}, \n"
                f"centers_q={sfcdc_centers_q}, centers_k={sfcdc_centers_k}, group_size={sfcdc_group_size}, \n"
                f"k0={sfcdc_k0}, k1={sfcdc_k1}, k2={sfcdc_k2}")
    replace_attention_processors(pipe, sfcdc_simulator)

    # --- Generate (use pipeline directly, all optimizations preserved) ---
    sfcdc_state.iter_idx = 0
    step_callback = _make_sfcdc_step_callback(sfcdc_simulator, num_inference_steps)
    video = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        use_dynamic_cfg=use_dynamic_cfg,
        generator=torch.Generator().manual_seed(seed),
        callback_on_step_end=step_callback,
    ).frames[0]

    export_to_video(video, output_path, fps=fps)
    logger.info(f"Video saved to {output_path}")


# ===========================================================================
#  CLI
# ===========================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="CogVideoX Text-to-Video with SFCDC sparse attention"
    )
    parser.add_argument("--prompt", type=str, default="A girl riding a bike.",
                        help="Text prompt for video generation.")
    parser.add_argument("--model_path", type=str, default="THUDM/CogVideoX1.5-5B",
                        help="Path to the pre-trained CogVideoX model.")
    parser.add_argument("--output_path", type=str, default="./output.mp4",
                        help="Output video file path.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames.")
    parser.add_argument("--width", type=int, default=None, help="Video width.")
    parser.add_argument("--height", type=int, default=None, help="Video height.")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps.")
    parser.add_argument("--guidance_scale", type=float, default=6.0,
                        help="Classifier-free guidance scale.")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16"], help="Computation dtype.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--fps", type=int, default=16, help="Output video FPS.")
    parser.add_argument("--use_dynamic_cfg", action="store_true", default=True,
                        help="Use dynamic CFG scheduling.")
    parser.add_argument("--prompt_path", type=str, default="./tests/CogVideoX/prompts",
                        help="Root directory for prompt files.")
    parser.add_argument("--prompt_id", type=int, default=None,
                        help="Load prompt from prompt_path/<id>/prompt.txt, overrides --prompt.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    generate_video(
        prompt=args.prompt,
        model_path=args.model_path,
        output_path=args.output_path,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        dtype=dtype,
        seed=args.seed,
        fps=args.fps,
        use_dynamic_cfg=args.use_dynamic_cfg,
        # SFCDC
        sfcdc_enabled=True,
        sfcdc_start_iter=15,
        sfcdc_end_iter=49,
        # sfcdc_centers_q=128*11,
        # sfcdc_centers_k=128*11,
        # sfcdc_group_size=1,
        sfcdc_centers_q=512*4,
        sfcdc_centers_k=512*4,
        sfcdc_group_size=3,
        sfcdc_k0=0.25,
        sfcdc_k1=0.1,
        sfcdc_k2=0.1,
        # Prompt system
        prompt_path=args.prompt_path,
        prompt_id=args.prompt_id,
    )
