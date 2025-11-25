# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae2_2 import Wan2_2_VAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.utils import best_output_size, masks_like

from datetime import datetime   # For MSE analysis logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import traceback

# helper: save latent-difference heatmap
def _save_latent_visual(diff, out_path, step_idx=None, mode='heatmap', vmax_percentile=99):
    """
    diff: numpy array or torch tensor.
        - If mode=='heatmap': expected shape [C*F, H*W] or [C, F, H, W] (will be flattened).
        - If mode=='tiles': expected shape [C, F, H, W].
    out_path: output file path.
    mode: 'heatmap' or 'tiles'
    """
    try:
        # convert to numpy
        if hasattr(diff, 'cpu'):
            diff = diff.detach().cpu().numpy()
        # if 4D [C,F,H,W] convert to 2D for heatmap or use directly for tiles
        if diff.ndim == 4:
            c, f, h, w = diff.shape
            if mode == 'heatmap':
                arr = diff.reshape(c * f, h * w)
            else:  # tiles
                tiles = diff
        elif diff.ndim == 2:
            # already [C*F, H*W]
            arr = diff
            # infer H,W if possible from out_path name? not possible -> require arr form for heatmap
            # for tiles mode, user should pass 4D.
        else:
            # flatten any other shape to heatmap
            arr = diff.reshape(diff.shape[0], -1)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if mode == 'heatmap':
            # compute vmax robustly
            abs_vals = np.abs(arr)
            vmax = np.percentile(abs_vals, vmax_percentile)
            if vmax == 0 or not np.isfinite(vmax):
                vmax = np.max(abs_vals) if np.max(abs_vals) > 0 else 1.0
            plt.figure(figsize=(12, 6))
            im = plt.imshow(arr, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto', interpolation='nearest')
            plt.colorbar(im, fraction=0.03, pad=0.03)
            if step_idx is not None:
                plt.title(f"latent diff step {step_idx}")
            plt.tight_layout()
            plt.savefig(out_path, bbox_inches='tight', dpi=150)
            plt.close()
        elif mode == 'tiles':
            # tiles mode: tiles shape [C, F, H, W] -> make list of images [C*F, H, W]
            if diff.ndim == 2:
                # cannot create tiles from 2D without H,W info
                raise ValueError("For tiles mode, provide diff as shape [C,F,H,W].")
            c, f, h, w = tiles.shape
            n = c * f
            imgs = tiles.reshape(n, h, w)
            # normalize per-tile for better visibility (optional: global normalize)
            # map to [0,1] for saving
            mins = imgs.min(axis=(1,2), keepdims=True)
            maxs = imgs.max(axis=(1,2), keepdims=True)
            denom = (maxs - mins)
            denom[denom == 0] = 1.0
            imgs_norm = (imgs - mins) / denom
            # determine grid size
            cols = int(math.ceil(math.sqrt(n)))
            rows = int(math.ceil(n / cols))
            # create big canvas
            canvas = np.zeros((rows * h, cols * w), dtype=np.float32)
            for idx in range(n):
                r = idx // cols
                cidx = idx % cols
                canvas[r*h:(r+1)*h, cidx*w:(cidx+1)*w] = imgs_norm[idx]
            plt.figure(figsize=(cols, rows))
            plt.imshow(canvas, cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
            if step_idx is not None:
                plt.title(f"latent tiles step {step_idx}")
            plt.tight_layout(pad=0)
            plt.savefig(out_path, bbox_inches='tight', dpi=150)
            plt.close()
        else:
            raise ValueError("mode should be 'heatmap' or 'tiles'")
    except Exception:
        try:
            plt.close()
        except Exception:
            pass
class WanTI2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_2_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model = self._configure_model(
            model=self.model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)

        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward, model)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def generate(self,
                 input_prompt,
                 img=None,
                 size=(1280, 704),
                 max_area=704 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            size (`tuple[int]`, *optional*, defaults to (1280,704)):
                Controls video resolution, (width,height).
            max_area (`int`, *optional*, defaults to 704*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # i2v
        if img is not None:
            return self.i2v(
                input_prompt=input_prompt,
                img=img,
                max_area=max_area,
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                n_prompt=n_prompt,
                seed=seed,
                offload_model=offload_model)
        # t2v
        return self.t2v(
            input_prompt=input_prompt,
            size=size,
            frame_num=frame_num,
            shift=shift,
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            n_prompt=n_prompt,
            seed=seed,
            offload_model=offload_model)

    def t2v(self,
            input_prompt,
            size=(1280, 704),
            frame_num=121,
            shift=5.0,
            sample_solver='unipc',
            sampling_steps=50,
            guide_scale=5.0,
            n_prompt="",
            seed=-1,
            offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (`tuple[int]`, *optional*, defaults to (1280,704)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 121):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync(),
        ):

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # save timesteps into JSON file for analysis
            timestep_log_dir = os.path.join("./attn_analysis/timestep", f"rank_{self.rank}")
            os.makedirs(timestep_log_dir, exist_ok=True)
            timestep_json_path = os.path.join(
                timestep_log_dir, f"timestep_values_rank{self.rank}.json")
            with open(timestep_json_path, "w") as json_file:
                json.dump({"timesteps": [t.item() for t in timesteps]}, json_file, indent=4)

            log_dir = "./log"
            viz_mode = 'heatmap'
            enable_log = False
            enable_pic = False
            time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            attn_save_enabled = True
            attn_score_save_enable = True
            attn_mask_out_dir = "./attn_analysis/attn_mask/top15_F29"
            attn_score_out_dir = "./attn_analysis/attn_score/top15_F29"
            attn_mask_param = 0.15
            attn_head = 0
            attn_mask_mode = "topk"
            attn_mask_format = 'npy'
            attn_rank_dir = None
            if attn_save_enabled:
                attn_mask_rank_dir = os.path.join(attn_mask_out_dir, f"rank_{self.rank}")
                os.makedirs(attn_mask_rank_dir, exist_ok=True)
                attn_score_rank_dir = os.path.join(attn_score_out_dir, f"rank_{self.rank}")
                os.makedirs(attn_score_rank_dir, exist_ok=True)
            
            log_rank_dir = os.path.join(log_dir, f"rank_{self.rank}")
            if enable_log:
                os.makedirs(log_rank_dir, exist_ok=True)
                log_path = os.path.join(log_rank_dir, f"MSE_anay_{time_stamp}.log")
                f_log = open(log_path, "a", buffering=1)
                f_log.write("time\tstep_idx\ttimestep\tmse\trel_mse\n")
                if enable_pic:
                    pic_root = os.path.join(log_rank_dir, f"pic_{viz_mode}_{time_stamp}")
                    os.makedirs(pic_root, exist_ok=True)
                else:
                    pic_root = None
            else:
                f_log = None
                pic_root = None

            # sample videos
            latents = noise
            mask1, mask2 = masks_like(noise, zero=False)

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            if offload_model or self.init_on_cpu:
                self.model.to(self.device)
                torch.cuda.empty_cache()

            self.model.reset_sparse_history(keys=("cond", "uncond"))

            # for _, t in enumerate(tqdm(timesteps)):
            for step_idx, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                temp_ts = torch.cat([
                    temp_ts,
                    temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
                ])
                timestep = temp_ts.unsqueeze(0)

                # 传入 attn_save 信息到 model.forward，让 model 内部每层 self_attn 保存 mask
                attn_save = None
                if attn_save_enabled:
                    attn_save = dict(
                        enable=True if (step_idx % 7 == 0) else False,
                        step=step_idx,
                        mask_out_dir=attn_mask_rank_dir,
                        param=attn_mask_param,
                        head=attn_head,
                        mode=attn_mask_mode,
                        rank=self.rank,
                        block_parts=target_shape[1],
                        out_format=attn_mask_format,
                        score_save_enable=attn_score_save_enable,
                        score_out_dir=attn_score_rank_dir,
                    )

                noise_pred_cond = self.model(
                    latent_model_input,
                    t=timestep,
                    attn_save=attn_save,
                    sparse_history_key="cond",
                    **arg_c,
                )[0]
                noise_pred_uncond = self.model(
                    latent_model_input,
                    t=timestep,
                    attn_save=None,
                    sparse_history_key="uncond",
                    **arg_null,
                )[0]

                # noise_pred_cond = self.model(
                #     latent_model_input, t=timestep, **arg_c)[0]
                # noise_pred_uncond = self.model(
                #     latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            # new: write to log file
                try:
                    a = latent_model_input[0].to(self.device)
                    b = latents[0].to(self.device)
                    diff = a - b
                    mse_tensor = (diff ** 2).mean()
                    denom = (a ** 2).mean()
                    if denom.item() > 0:
                        rel_mse_tensor = mse_tensor / denom
                    else:
                        rel_mse_tensor = torch.tensor(float('nan'), device=self.device)
                    mse = float(mse_tensor.detach().cpu().item())
                    rel_mse = float(rel_mse_tensor.detach().cpu().item())
                    # transform differences to [C*F, H*W]
                    try:
                        # c, f, h, w = diff.shape
                        diff_np = diff.detach().cpu().numpy()
                        # only rank 0 saves pic
                        if pic_root is not None:
                            out_path = os.path.join(pic_root, f"latent_diff_step{step_idx}.png")
                            _save_latent_visual(diff_np, out_path, step_idx=step_idx, mode=viz_mode)
                    except Exception:
                        pass
                    del a, b, diff, mse_tensor, rel_mse_tensor
                except Exception:
                    mse = rel_mse = float('nan')
                # only rank 0 writes log
                if f_log is not None:
                    f_log.write(f"{datetime.now().isoformat()}\t{step_idx}\t{t}\t{mse:.8e}\t{rel_mse:.8e}\n")
            if f_log is not None:
                f_log.close()
            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            # if self.rank == 0:
            #     videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
        torch.cuda.empty_cache()
        videos = self.vae.decode(x0)
        return videos[0] if videos is not None else None

    def i2v(self,
            input_prompt,
            img,
            max_area=704 * 1280,
            frame_num=121,
            shift=5.0,
            sample_solver='unipc',
            sampling_steps=40,
            guide_scale=5.0,
            n_prompt="",
            seed=-1,
            offload_model=True):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 704*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 121):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (121)
                - H: Frame height (from max_area)
                - W: Frame width (from max_area)
        """
        # preprocess
        ih, iw = img.height, img.width
        dh, dw = self.patch_size[1] * self.vae_stride[1], self.patch_size[
            2] * self.vae_stride[2]
        ow, oh = best_output_size(iw, ih, dw, dh, max_area)

        scale = max(ow / iw, oh / ih)
        img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)

        # center-crop
        x1 = (img.width - ow) // 2
        y1 = (img.height - oh) // 2
        img = img.crop((x1, y1, x1 + ow, y1 + oh))
        assert img.width == ow and img.height == oh

        # to tensor
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device).unsqueeze(1)

        F = frame_num
        seq_len = ((F - 1) // self.vae_stride[0] + 1) * (
            oh // self.vae_stride[1]) * (ow // self.vae_stride[2]) // (
                self.patch_size[1] * self.patch_size[2])
        seq_len = int(math.ceil(seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
            oh // self.vae_stride[1],
            ow // self.vae_stride[2],
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        z = self.vae.encode([img])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync(),
        ):

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latent = noise
            mask1, mask2 = masks_like([noise], zero=True)
            latent = (1. - mask2[0]) * z[0] + mask2[0] * latent

            arg_c = {
                'context': [context[0]],
                'seq_len': seq_len,
            }

            arg_null = {
                'context': context_null,
                'seq_len': seq_len,
            }

            if offload_model or self.init_on_cpu:
                self.model.to(self.device)
                torch.cuda.empty_cache()

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(self.device)]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                temp_ts = torch.cat([
                    temp_ts,
                    temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
                ])
                timestep = temp_ts.unsqueeze(0)

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)
                latent = (1. - mask2[0]) * z[0] + mask2[0] * latent

                x0 = [latent]
                del latent_model_input, timestep

            if offload_model:
                self.model.cpu()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latent, x0
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
