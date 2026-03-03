import os
import sys

import argparse
import logging
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
import torchvision
from PIL import Image
import imageio

# Setup paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
# 1. Add 'src' to path so 'SFCDC' and 'SVG2' can be imported by 'wan' modules
sys.path.append(os.path.dirname(current_dir))
# 2. Add 'src/Wan2.1' to path so 'wan' can be imported directly
sys.path.append(os.path.join(os.path.dirname(current_dir), 'Wan2.1'))

import wan
from wan.text2video_sfcdc import WanT2V as WanT2V_SFCDC
from wan.image2video_sfcdc import WanI2V as WanI2V_SFCDC
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import str2bool


T2V_CKPT_DIR = "./Wan2.1/model/Wan2.1-T2V-14B"
I2V_CKPT_DIR = "./Wan2.1/model/Wan2.1-I2V-14B-720P"


EXAMPLE_PROMPT = {
    "t2v-14B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "i2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
}


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
        args.image = EXAMPLE_PROMPT[args.task]["image"]

    if args.task == "i2v-14B":
        assert args.image is not None, "Please specify the image path for i2v."

    # The default sampling steps are 40 for i2v and 50 for t2v
    if args.sample_steps is None:
        args.sample_steps = 50
        if "i2v" in args.task:
            args.sample_steps = 40

    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0

    if args.frame_num is None:
        args.frame_num = 81

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)

    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _load_prompt_assets_from_id(prompt_path, prompt_id, prompt_filename, image_filename):
    prompt_dir = os.path.join(prompt_path, str(prompt_id))
    if not os.path.isdir(prompt_dir):
        raise FileNotFoundError(f"Prompt directory not found: {prompt_dir}")

    prompt_file_path = os.path.join(prompt_dir, prompt_filename)
    if not os.path.isfile(prompt_file_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file_path}")

    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        prompt = f.read().strip()

    image_path = None
    for ext in ['png', 'jpg', 'jpeg', 'webp']:
        potential_path = os.path.join(prompt_dir, f'{image_filename}.{ext}')
        if os.path.exists(potential_path):
            image_path = potential_path
            break

    return prompt, image_path


def _get_all_prompt_ids(prompt_path):
    if not os.path.isdir(prompt_path):
        raise FileNotFoundError(f"Prompt path not found: {prompt_path}")

    numeric_ids = []
    for entry in os.listdir(prompt_path):
        entry_path = os.path.join(prompt_path, entry)
        if os.path.isdir(entry_path) and entry.isdigit():
            numeric_ids.append(int(entry))

    return sorted(numeric_ids)


def _choose_task_and_ckpt(image_path):
    if image_path is None:
        return "t2v-14B", T2V_CKPT_DIR
    return "i2v-14B", I2V_CKPT_DIR


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a video from text/image using Wan2.1 + SFCDC acceleration"
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="./tests/Wan2.2/prompts",
        help="The root directory for prompts. Each prompt should be in a subdirectory named with its prompt_id, containing a text file for the prompt and optionally an image file."
    )
    parser.add_argument(
        "--prompt_id",
        type=int,
        default=None,
        help="ID of the prompt to load from the './prompts' directory. This will override --prompt and --image."
    )
    parser.add_argument(
        "--all_id",
        action="store_true",
        default=False,
        help="When enabled, ignore --prompt_id and run all numeric subdirectories under --prompt_path."
    )
    parser.add_argument(
        "--all_video_path",
        type=str,
        default=None,
        help="Output directory for all generated videos when --all_id is enabled."
    )
    parser.add_argument(
        "--prompt_filename",
        type=str,
        default="prompt.txt",
        help="The filename for the prompt text file within the prompt_id directory."
    )
    parser.add_argument(
        "--image_filename",
        type=str,
        default="pic",
        help="The base filename for the image within the prompt_id directory (extension will be auto-detected)."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14B",
        choices=["t2v-14B", "t2v-1.3B", "i2v-14B"],
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video.")
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames to sample. The number should be 4n+1")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the Wan2.1 checkpoint directory. In prompt_id/all_id mode, it will be auto-selected by model type.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each forward.")
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to.")
    parser.add_argument(
        "--overlap",
        action="store_true",
        default=False,
        help="Whether to overwrite an existing output file. If disabled, existing file will be skipped."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from (for i2v).")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.")

    args = parser.parse_args()

    torch.cuda.manual_seed_all(42)

    return args


def save_video(tensor, save_file, fps=16, value_range=(-1, 1)):
    """
    Save a video tensor to file. Replacement for cache_video that avoids
    the dtype bug in the original retry loop (uint8 tensor on retry).
    
    Args:
        tensor: [C, T, H, W] float tensor
        save_file: output file path
        fps: frames per second
        value_range: (min, max) for normalization
    """
    # Ensure float32 for safe arithmetic
    tensor = tensor.detach().float().cpu()
    # Normalize from value_range to [0, 1]
    low, high = value_range
    tensor = tensor.clamp(low, high)
    tensor = (tensor - low) / max(high - low, 1e-5)
    # Convert to uint8 [T, H, W, C]
    tensor = (tensor * 255).to(torch.uint8)
    # [C, T, H, W] -> [T, H, W, C]
    frames = tensor.permute(1, 2, 3, 0).numpy()

    writer = imageio.get_writer(save_file, fps=fps, codec='libx264', quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), "t5_fsdp and dit_fsdp are not supported in non-distributed environments."

    cfg = WAN_CONFIGS[args.task]

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info(f"Input prompt: {args.prompt}")
    img = None
    if args.image is not None:
        img = Image.open(args.image).convert("RGB")
        logging.info(f"Input image: {args.image}")

    # prompt extend
    if args.use_prompt_extend:
        logging.info("Extending prompt ...")
        if rank == 0:
            if args.prompt_extend_method == "dashscope":
                prompt_expander = DashScopePromptExpander(
                    model_name=args.prompt_extend_model,
                    task=args.task,
                    is_vl=args.image is not None)
            elif args.prompt_extend_method == "local_qwen":
                prompt_expander = QwenPromptExpander(
                    model_name=args.prompt_extend_model,
                    task=args.task,
                    is_vl=args.image is not None,
                    device=rank)
            else:
                raise NotImplementedError(
                    f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

            prompt_output = prompt_expander(
                args.prompt,
                image=img,
                tar_lang=args.prompt_extend_target_lang,
                seed=args.base_seed)
            if prompt_output.status == False:
                logging.info(
                    f"Extending prompt failed: {prompt_output.message}")
                logging.info("Falling back to original prompt.")
                input_prompt = args.prompt
            else:
                input_prompt = prompt_output.prompt
            input_prompt = [input_prompt]
        else:
            input_prompt = [None]
        if dist.is_initialized():
            dist.broadcast_object_list(input_prompt, src=0)
        args.prompt = input_prompt[0]
        logging.info(f"Extended prompt: {args.prompt}")

    if "t2v" in args.task:
        logging.info("Creating WanT2V pipeline (SFCDC).")
        wan_t2v = WanT2V_SFCDC(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )

        logging.info(f"Generating video with SFCDC acceleration...")
        video = wan_t2v.generate(
            args.prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)
    elif "i2v" in args.task:
        logging.info("Creating WanI2V pipeline (SFCDC).")
        wan_i2v = WanI2V_SFCDC(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )

        logging.info(f"Generating video with SFCDC acceleration...")
        video = wan_i2v.generate(
            args.prompt,
            img,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)
    else:
        raise NotImplementedError(f"Unsupported task for SFCDC: {args.task}")

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                     "_")[:50]
            suffix = '.mp4'
            args.save_file = f"sfcdc_{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{formatted_prompt}_{formatted_time}" + suffix

        logging.info(f"Saving generated video to {args.save_file}")
        save_video(
            tensor=video,
            save_file=args.save_file,
            fps=cfg.sample_fps,
            value_range=(-1, 1))
    del video

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()

    if args.all_id:
        if args.all_video_path is None:
            raise ValueError("Please specify --all_video_path when --all_id is enabled.")

        if args.prompt_id is not None:
            print("--all_id is enabled, ignoring --prompt_id.")

        prompt_ids = _get_all_prompt_ids(args.prompt_path)
        if len(prompt_ids) == 0:
            raise ValueError(f"No numeric prompt directories found under: {args.prompt_path}")

        os.makedirs(args.all_video_path, exist_ok=True)

        for prompt_id in prompt_ids:
            run_args = argparse.Namespace(**vars(args))
            run_args.prompt_id = prompt_id
            run_args.prompt, run_args.image = _load_prompt_assets_from_id(
                prompt_path=run_args.prompt_path,
                prompt_id=prompt_id,
                prompt_filename=run_args.prompt_filename,
                image_filename=run_args.image_filename,
            )
            run_args.task, run_args.ckpt_dir = _choose_task_and_ckpt(run_args.image)
            run_args.save_file = os.path.join(run_args.all_video_path, f"{prompt_id}.mp4")

            if (not run_args.overlap) and os.path.exists(run_args.save_file):
                print(f"Skip prompt_id={prompt_id}, file already exists: {run_args.save_file}")
                continue

            print(f"Loaded prompt for id={prompt_id}, task={run_args.task}")
            if run_args.image is not None:
                print(f"Loaded image from {run_args.image}")

            _validate_args(run_args)
            generate(run_args)
    else:
        if args.prompt_id is not None:
            args.prompt, args.image = _load_prompt_assets_from_id(
                prompt_path=args.prompt_path,
                prompt_id=args.prompt_id,
                prompt_filename=args.prompt_filename,
                image_filename=args.image_filename,
            )
            args.task, args.ckpt_dir = _choose_task_and_ckpt(args.image)
            print(f"Loaded prompt for id={args.prompt_id}, task={args.task}")
            if args.image is not None:
                print(f"Loaded image from {args.image}")

        if (args.save_file is not None) and (not args.overlap) and os.path.exists(args.save_file):
            print(f"Skip generation, file already exists: {args.save_file}")
            sys.exit(0)

        _validate_args(args)
        generate(args)
