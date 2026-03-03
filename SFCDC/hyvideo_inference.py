import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

# === Path setup ===
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)
sys.path.insert(0, os.path.join(src_dir, "HunyuanVideo"))

from SFCDC.sfcdc_v5 import SFCDC_Simulator

from hyvideo.inference import HunyuanVideoSampler
from hyvideo.modules.attenion import sfcdc_state
from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.utils.data_utils import align_to

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _is_kernel_or_debug_arg_sequence(unknown_args):
    """Return True when unknown args are likely injected by notebook/debug runtime."""
    if not unknown_args:
        return False

    # Common ipykernel pattern: "-f <kernel-connection-file>.json"
    if len(unknown_args) == 2 and unknown_args[0] == "-f" and unknown_args[1].endswith(".json"):
        return True

    # Other common jupyter runtime flags
    kernel_prefixes = (
        "--ip=",
        "--stdin=",
        "--control=",
        "--hb=",
        "--shell=",
        "--iopub=",
        "--transport=",
        "--Session.",
    )
    return all(arg.startswith(kernel_prefixes) for arg in unknown_args)


# ===========================================================================
#  Compute Latent Geometry
# ===========================================================================
def compute_latent_geometry(vae_name: str, video_length: int, height: int, width: int):
    """
    Auto-compute the latent video token geometry (l_f, l_h, l_w) for SFCDC.

    HunyuanVideo uses:
      - VAE "884": temporal compress (t-1)//4+1, spatial compress //8
      - patch_size = [1, 2, 2]
    So the final token grid is:
      - l_f = (video_length - 1) // 4 + 1    (for 884 VAE, then // patch_t=1)
      - l_h = height // 8 // 2 = height // 16
      - l_w = width  // 8 // 2 = width  // 16

    Returns:
        (l_f, l_h, l_w)
    """
    # VAE temporal/spatial compression
    if "884" in vae_name:
        latent_f = (video_length - 1) // 4 + 1
        latent_h = height // 8
        latent_w = width // 8
    elif "888" in vae_name:
        latent_f = (video_length - 1) // 8 + 1
        latent_h = height // 8
        latent_w = width // 8
    else:
        latent_f = video_length
        latent_h = height // 8
        latent_w = width // 8

    # Patch embedding: patch_size = [1, 2, 2]
    patch_t, patch_h, patch_w = 1, 2, 2
    l_f = latent_f // patch_t
    l_h = latent_h // patch_h
    l_w = latent_w // patch_w

    n_video_tokens = l_f * l_h * l_w
    logger.info(
        f"Latent geometry: l_f={l_f}, l_h={l_h}, l_w={l_w}, "
        f"video_tokens={n_video_tokens}"
    )
    return l_f, l_h, l_w


# ===========================================================================
#  Step Callback
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


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _load_prompt_from_id(prompt_path: str, prompt_id: int, prompt_filename: str):
    prompt_file = os.path.join(prompt_path, str(prompt_id), prompt_filename)
    if not os.path.isfile(prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read().strip()


def _get_all_prompt_ids(prompt_path: str):
    if not os.path.isdir(prompt_path):
        raise FileNotFoundError(f"Prompt path not found: {prompt_path}")

    prompt_ids = []
    for entry in os.listdir(prompt_path):
        entry_path = os.path.join(prompt_path, entry)
        if os.path.isdir(entry_path) and entry.isdigit():
            prompt_ids.append(int(entry))
    return sorted(prompt_ids)


def _build_output_filenames(prompt_id: Optional[int], num_videos: int):
    if prompt_id is None:
        return None
    if num_videos <= 1:
        return [f"{prompt_id}.mp4"]
    return [f"{prompt_id}_{i}.mp4" for i in range(num_videos)]


def add_prompt_id_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Prompt ID args")

    group.add_argument(
        "--prompt-path", "--prompt_path", dest="prompt_path", type=str, default="./tests/HunyuanVideo/prompts",
        help="Root directory for prompt files.",
    )
    group.add_argument(
        "--prompt-id", "--prompt_id", dest="prompt_id", type=int, default=None,
        help="Load prompt from <prompt-path>/<id>/prompt.txt, overrides --prompt.",
    )
    group.add_argument(
        "--prompt-filename", "--prompt_filename", dest="prompt_filename", type=str, default="prompt.txt",
        help="Prompt text filename under each prompt id directory.",
    )
    group.add_argument(
        "--all-id", "--all_id", dest="all_id", action="store_true", default=False,
        help="Run all numeric prompt directories under --prompt-path.",
    )
    group.add_argument(
        "--all-video-path", "--all_video_path", dest="all_video_path", type=str, default=None,
        help="Output directory used when --all-id is enabled.",
    )
    parser.add_argument(
        "--overlap",
        action="store_true",
        default=False,
        help="Whether to overwrite an existing output file. If disabled, existing file will be skipped."
    )

    return parser


def build_parser():
    """
    Build a unified argument parser that includes:
      1. HunyuanVideo native args (network, VAE, text encoder, denoise, inference)
      2. SFCDC-specific args
      3. Prompt ID system args
    """
    # Start from HunyuanVideo's parser (it returns argparse.ArgumentParser)
    # We call parse_hyvideo_args() internals manually to extend the parser.
    parser = argparse.ArgumentParser(
        description="HunyuanVideo T2V with SFCDC sparse attention"
    )

    # === HunyuanVideo native argument groups ===
    # We import from hyvideo.config the individual add_*_args functions
    from hyvideo.config import (
        add_network_args,
        add_extra_models_args,
        add_denoise_schedule_args,
        add_inference_args,
        add_parallel_args,
    )
    add_network_args(parser)
    add_extra_models_args(parser)
    add_denoise_schedule_args(parser)
    add_inference_args(parser)
    add_parallel_args(parser)

    # === Prompt ID ===
    add_prompt_id_args(parser)

    return parser


# ===========================================================================
#  Main
# ===========================================================================
def main():
    parser = build_parser()
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        if _is_kernel_or_debug_arg_sequence(unknown_args):
            logger.warning(
                "Ignored runtime-injected args: %s", " ".join(unknown_args)
            )
        else:
            parser.error(f"Unrecognized arguments: {' '.join(unknown_args)}")

    # --- Sanity check from HunyuanVideo ---
    from hyvideo.config import sanity_check_args
    args = sanity_check_args(args)

    # --- Resolve prompt jobs (single or batch) ---
    jobs = []
    if args.all_id:
        if args.all_video_path is None:
            raise ValueError("Must provide --all-video-path when --all-id is enabled.")
        prompt_ids = _get_all_prompt_ids(args.prompt_path)
        if len(prompt_ids) == 0:
            raise ValueError(f"No numeric prompt directories found under: {args.prompt_path}")
        for prompt_id in prompt_ids:
            prompt = _load_prompt_from_id(args.prompt_path, prompt_id, args.prompt_filename)
            jobs.append((prompt_id, prompt))
        logger.info(f"Batch mode enabled: {len(jobs)} prompts loaded from {args.prompt_path}")
    else:
        if args.prompt_id is not None:
            args.prompt = _load_prompt_from_id(args.prompt_path, args.prompt_id, args.prompt_filename)
            logger.info(f"Loaded prompt for id={args.prompt_id}")

        if args.prompt is None:
            raise ValueError("Must provide --prompt / --prompt-id, or enable --all-id.")
        jobs.append((args.prompt_id, args.prompt))

    logger.info(
        f"Video config: size={args.video_size}, length={args.video_length}, "
        f"steps={args.infer_steps}, cfg={args.cfg_scale}, "
        f"embedded_cfg={args.embedded_cfg_scale}, jobs={len(jobs)}"
    )

    # =====================================================================
    # Load models via HunyuanVideoSampler
    # =====================================================================
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`model-base` does not exist: {models_root_path}")

    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(
        models_root_path, args=args
    )
    # from_pretrained may update args (e.g. latent_channels)
    args = hunyuan_video_sampler.args

    # =====================================================================
    # Setup SFCDC
    # =====================================================================
    height, width = args.video_size[0], args.video_size[1]
    video_length = args.video_length

    # Keep geometry aligned with the actual size used in sampler.predict()
    target_height = align_to(height, 16)
    target_width = align_to(width, 16)

    l_f, l_h, l_w = compute_latent_geometry(
        vae_name=args.vae,
        video_length=video_length,
        height=target_height,
        width=target_width,
    )

    # HunyuanVideo's text_len (default 256)
    text_token_length = args.text_len

    sfcdc_simulator = SFCDC_Simulator(
        enabled=True,
        clusbegin_iter=14,
        start_iter=15,
        end_iter=49,
        centers_q=256*11,
        centers_k=256*11,
        group_size=3,
        k0=0.25,
        k1=0.1,
        k2=0.1,
        l_f=l_f,    # (129 - 1) / 4 + 1 = 33 for 884 VAE
        l_h=l_h,    # 720 / 8 / 2 = 45 for 884 VAE
        l_w=l_w,    # 1280 / 8 / 2 = 80 for 884 VAE
        text_token_length=text_token_length,
        text_location="end",   # HunyuanVideo: [img_tokens, txt_tokens]
    )

    # Inject SFCDC into the global state read by models.py
    sfcdc_state.simulator = sfcdc_simulator
    sfcdc_state.iter_idx = 0

    logger.info(
        f"SFCDC: enabled={sfcdc_simulator.enabled}, "
        f"text_tokens={text_token_length}, text_location=end, "
        f"l_f={l_f}, l_h={l_h}, l_w={l_w}, "
        f"target_size=({target_height}, {target_width}), "
        f"start_iter={sfcdc_simulator.start_iter}, end_iter={sfcdc_simulator.end_iter}"
    )

    # =====================================================================
    # Create save directory
    # =====================================================================
    save_root = args.all_video_path if args.all_id else args.save_path
    if (not args.all_id) and args.save_path_suffix:
        save_root = f"{save_root}_{args.save_path_suffix}"
    os.makedirs(save_root, exist_ok=True)

    # =====================================================================
    # Generate
    # =====================================================================
    step_callback = _make_sfcdc_step_callback(sfcdc_simulator, args.infer_steps)
    for job_index, (prompt_id, prompt_text) in enumerate(jobs, start=1):
        logger.info(f"[{job_index}/{len(jobs)}] Generating for prompt_id={prompt_id}")
        sfcdc_state.iter_idx = 0

        planned_filenames = _build_output_filenames(prompt_id, args.num_videos)
        if (not args.overlap) and planned_filenames is not None:
            planned_paths = [os.path.join(save_root, name) for name in planned_filenames]
            if all(os.path.exists(path) for path in planned_paths):
                logger.info(
                    f"Skip inference for prompt_id={prompt_id}: all target files already exist."
                )
                continue

        outputs = hunyuan_video_sampler.predict(
            prompt=prompt_text,
            height=height,
            width=width,
            video_length=video_length,
            seed=args.seed,
            negative_prompt=args.neg_prompt,
            infer_steps=args.infer_steps,
            guidance_scale=args.cfg_scale,
            num_videos_per_prompt=args.num_videos,
            flow_shift=args.flow_shift,
            batch_size=args.batch_size,
            embedded_guidance_scale=args.embedded_cfg_scale,
            callback_on_step_end=step_callback if sfcdc_simulator.enabled else None,
        )

        samples = outputs["samples"]

        if "LOCAL_RANK" not in os.environ or int(os.environ["LOCAL_RANK"]) == 0:
            for i, sample in enumerate(samples):
                sample = sample.unsqueeze(0)
                if prompt_id is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"sample_{timestamp}_{i}.mp4"
                elif len(samples) == 1:
                    filename = f"{prompt_id}.mp4"
                else:
                    filename = f"{prompt_id}_{i}.mp4"

                cur_save_path = os.path.join(save_root, filename)
                if (not args.overlap) and os.path.exists(cur_save_path):
                    logger.info(f"Skip existing file: {cur_save_path}")
                    continue
                save_videos_grid(sample, cur_save_path, fps=24)
                logger.info(f"Video saved to: {cur_save_path}")


if __name__ == "__main__":
    main()
