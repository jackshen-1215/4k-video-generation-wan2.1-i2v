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

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.clip import CLIPModel
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


def get_dimension_slices_and_sizes(begin, end, size, wrap=True):
    if not wrap:
        begin_clamped = max(0, begin)
        end_clamped = min(size, end)
        if begin_clamped >= end_clamped:
            return [], []
        return [slice(begin_clamped, end_clamped)], [end_clamped - begin_clamped]

    slices = []
    sizes = [] 
    current_pos = begin
    
    while current_pos < end:
        start_idx = current_pos % size # Start index in the current tile
        next_boundary = ((current_pos // size) + 1) * size # The start position of the next tile
        end_pos = min(end, next_boundary) # The end position of the current tile
        length = end_pos - current_pos
        end_idx = (start_idx + length) % size # End index in the current tile

        if end_idx > start_idx:
            slices.append(slice(start_idx, end_idx))
            sizes.append(end_idx - start_idx)
        else: 
            slices.append(slice(start_idx, size))
            sizes.append(size - start_idx)
            if end_idx > 0:
                slices.append(slice(0, end_idx))
                sizes.append(end_idx)
        current_pos = end_pos
    
    return slices, sizes
        
            
class RingLatent2D:
    def __init__(self, latent_tensor, wrap=True):
        self.torch_latent = latent_tensor.clone()
        self.wrap = wrap
        self.batch_size = self.torch_latent.shape[0] 
        self.num_frames = self.torch_latent.shape[1]
        self.channels = self.torch_latent.shape[2]
        self.height = self.torch_latent.shape[3]
        self.width = self.torch_latent.shape[4]

    def get_shape(self):
        return self.torch_latent.shape
    
    def get_window_latent(self, top: int = None, bottom: int = None, left: int = None, right: int = None):
        if top is None:
            top = 0
        if bottom is None:
            bottom = self.height
        if left is None:
            left = 0
        if right is None:
            right = self.width
            
        # Ensure the indices are within the valid range
        if self.wrap:
            assert 0 <= top < bottom <= self.height * 2, f"Invalid top {top} and bottom {bottom}"
            assert 0 <= left < right <= self.width * 2, f"Invalid left {left} and right {right}"
        
        height_slices, height_sizes = get_dimension_slices_and_sizes(top, bottom, self.height, self.wrap)
        width_slices, width_sizes = get_dimension_slices_and_sizes(left, right, self.width, self.wrap)

        # Get the parts of the latent tensor
        parts = []
        for h_slice in height_slices:
            row_parts = []
            for w_slice in width_slices:
                part = self.torch_latent[:, :, :, h_slice, w_slice]
                row_parts.append(part)
            row = torch.cat(row_parts, dim=4)
            parts.append(row)
        desired_latent = torch.cat(parts, dim=3)
        
        return desired_latent.clone()
    
    def set_window_latent(self, input_latent: torch.Tensor,
                          top: int = None,
                          bottom: int = None,
                          left: int = None,
                          right: int = None):
        if top is None:
            top = 0
        if bottom is None:
            bottom = self.height
        if left is None:
            left = 0
        if right is None:
            right = self.width

        assert 0 <= top < bottom <= self.height * 2, f"Invalid top {top} and bottom {bottom}"
        assert 0 <= left < right <= self.width * 2, f"Invalid left {left} and right {right}"
        assert bottom - top <= self.height, f"warp should not occur"
        assert right - left <= self.width, f"warp should not occur"

       # Calculate the target latent tensor
        target_height = bottom - top if bottom <= self.height else (self.height - top) + (bottom % self.height)
        target_width = right - left if right <= self.width else (self.width - left) + (right % self.width)

        width_slices, width_sizes = get_dimension_slices_and_sizes(left, right, self.width, self.wrap)
        height_slices, height_sizes = get_dimension_slices_and_sizes(top, bottom, self.height, self.wrap)

        target_height = sum(height_sizes)
        target_width = sum(width_sizes)
        # Check the shape of the input latent tensor
        assert input_latent.shape[3:] == (target_height, target_width), f"Input latent shape {input_latent.shape[3:]} does not match target window shape {(target_height, target_width)}"

        # Write the parts of the latent tensor
        h_start = 0
        for h_slice, h_size in zip(height_slices, height_sizes):
            w_start = 0
            for w_slice, w_size in zip(width_slices, width_sizes):
                input_part = input_latent[:, :, :, h_start:h_start+h_size, w_start:w_start+w_size]
                self.torch_latent[:, :, :, h_slice, w_slice] = input_part
                w_start += w_size
            h_start += h_size

# Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)

class WanI2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        init_on_cpu=True,
    ):
        r"""
        Initializes the image-to-video generation model components.

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
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir,
                                         config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if t5_fsdp or dit_fsdp or use_usp:
            init_on_cpu = False

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            if not init_on_cpu:
                self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def _generate_coarse_video(self,
                 input_prompt,
                 img,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=10,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h_img, w_img = img_tensor.shape[1:]
        aspect_ratio = h_img / w_img
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16, (F - 1) // 4 + 1,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        F_model_sched = (F - 1) // self.config.vae_stride[0] + 1
        
        C_mask = 4
        msk_cond = torch.ones(1, 1, lat_h, lat_w, device=self.device)
        if F_model_sched > 1:
            msk_zeros = torch.zeros(1, F_model_sched - 1, lat_h, lat_w, device=self.device)
            msk_cond_frames = torch.cat([msk_cond, msk_zeros], dim=1)
        else:
            msk_cond_frames = msk_cond

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

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

        self.clip.model.to(self.device)
        clip_context = self.clip.visual([img_tensor[:, None, :, :]])
        if offload_model:
            self.clip.model.cpu()

        vae_input_img_part = torch.nn.functional.interpolate(
            img_tensor[None], size=(h, w), mode='bicubic'
        ).squeeze(0)

        if F > 1:
            vae_input_zeros_part = torch.zeros(vae_input_img_part.shape[0], F - 1, h, w,
                                                dtype=vae_input_img_part.dtype, device=self.device)
            vae_input_full = torch.cat([vae_input_img_part.unsqueeze(1), vae_input_zeros_part], dim=1)
        else:
            vae_input_full = vae_input_img_part.unsqueeze(1)
        
        y = self.vae.encode([vae_input_full])[0]
        
        msk_final_channels = 4
        msk_y_cond_part = torch.zeros(msk_final_channels, F_model_sched, lat_h, lat_w, device=self.device)
        if F_model_sched > 0:
            msk_y_cond_part[:, 0, :, :] = 1
        
        y_condition = torch.cat([msk_y_cond_part, y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

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
            
            latent = noise
            denoised_x0 = None
            
            arg_c = {
                'context': context,
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y_condition],
            }

            arg_null = {
                'context': context_null,
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y_condition],
            }

            if offload_model:
                torch.cuda.empty_cache()

            self.model.to(self.device)
            for i_step, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(self.device)]
                timestep = torch.tensor([t], device=self.device)

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

                step_output = sample_scheduler.step(
                    noise_pred,
                    t,
                    latent,
                    return_dict=True,
                    generator=seed_g)
                
                latent = step_output.prev_sample
                if hasattr(step_output, 'pred_original_sample'):
                    denoised_x0 = step_output.pred_original_sample

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            
            videos = self.vae.decode([denoised_x0 if denoised_x0 is not None else latent])

        del noise, latent, y_condition, context, context_null, clip_context, sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
        
        return videos[0] if self.rank == 0 else None

    def re_noise(self, x_start, noise, step_a, step_b, scheduler):
        timestep_a = scheduler.timesteps[step_a]
        timestep_b = scheduler.timesteps[step_b]

        alpha_cumprod = scheduler.alphas_cumprod.to(x_start.device)

        alpha_cumprod_a = alpha_cumprod[timestep_a]
        alpha_cumprod_b = alpha_cumprod[timestep_b]

        c = torch.sqrt(alpha_cumprod_b / alpha_cumprod_a)
        s = torch.sqrt(1 - alpha_cumprod_b / alpha_cumprod_a)

        x_b = c * x_start + s * noise

        return x_b
    
    def _decode_tiled(self, latent_tensor, tile_resolution_str="832x480"):
        # Parse tile resolution
        tile_w_pixels, tile_h_pixels = map(int, tile_resolution_str.split('x'))
        
        # Convert pixel tile size to latent tile size
        patch_h_model, patch_w_model = self.model.patch_size[1], self.model.patch_size[2]
        tile_h_lat = (tile_h_pixels // self.vae_stride[1] // patch_h_model) * patch_h_model
        tile_w_lat = (tile_w_pixels // self.vae_stride[2] // patch_w_model) * patch_w_model
        
        # Get full latent dimensions
        full_h_lat, full_w_lat = latent_tensor.shape[-2:]
        
        # Create a buffer for the full output video
        full_h_pixels = full_h_lat * self.vae_stride[1]
        full_w_pixels = full_w_lat * self.vae_stride[2]
        # Calculate video frames from latent frames using temporal stride
        latent_frames = latent_tensor.shape[1]
        video_frames = (latent_frames - 1) * self.vae_stride[0] + 1
        output_video = torch.zeros(
            3, # RGB channels
            video_frames, 
            full_h_pixels, 
            full_w_pixels, 
            device=self.device,
            dtype=self.param_dtype
        )
        
        # Loop through tiles
        for h_start in range(0, full_h_lat, tile_h_lat):
            for w_start in range(0, full_w_lat, tile_w_lat):
                h_end = min(h_start + tile_h_lat, full_h_lat)
                w_end = min(w_start + tile_w_lat, full_w_lat)
                
                logging.info(f"Decoding tile at latent coordinates H:{h_start}-{h_end}, W:{w_start}-{w_end}")
                
                # Extract the latent tile
                latent_tile = latent_tensor[:, :, h_start:h_end, w_start:w_end]
                
                # VAE expects a list of tensors
                decoded_tile_list = self.vae.decode([latent_tile])
                if not decoded_tile_list:
                    continue
                
                decoded_tile = decoded_tile_list[0]
                
                # Place the decoded tile into the full output buffer
                h_start_pixels = h_start * self.vae_stride[1]
                w_start_pixels = w_start * self.vae_stride[2]
                
                # Get the actual shape of the decoded tile to handle edge cases
                _, _, h_decoded, w_decoded = decoded_tile.shape
                
                output_video[:, :, h_start_pixels:h_start_pixels+h_decoded, w_start_pixels:w_start_pixels+w_decoded] = decoded_tile
        
        return [output_video]

    def generate(self,
                 input_prompt,
                 img,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 enable_tiling: bool = True,
                 tile_window_height_factor: float = 1.0,
                 tile_window_width_factor: float = 0.5,
                 tile_stride_factor: float = 0.25,
                 use_coarse_to_fine: bool = True,
                 coarse_max_area_factor: float = 0.25,
                 coarse_sampling_steps: int = 10,
                 guidance_scale_schedule: tuple = (0.01, 0.01),
                 is_panorama: bool = False,
                 tile_resolution: str | None = None,
                ):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            enable_tiling (`bool`, *optional*, defaults to True):
                Whether to use tile-based denoising.
            tile_window_height_factor (`float`, *optional*, defaults to 1.0):
                Tile height as a factor of the full latent height.
            tile_window_width_factor (`float`, *optional*, defaults to 0.5):
                Tile width as a factor of the full latent width.
            tile_stride_factor (`float`, *optional*, defaults to 0.25):
                Stride for sliding window as a factor of the tile window dimension.
                A lower value means more overlap and more tiles.
            use_coarse_to_fine (`bool`, *optional*, defaults to False):
                Enable two-stage generation where a low-res video guides a high-res one.
            coarse_max_area_factor (`float`, *optional*, defaults to 0.25):
                The factor to reduce `max_area` by for the initial coarse video generation. This is ignored if the coarse video is generated at a fixed 480p.
            coarse_sampling_steps (`int`, *optional*, defaults to 10):
                Number of sampling steps for the coarse video generation.
            guidance_scale_schedule (`tuple`, *optional*, defaults to (0.0, 0.3)):
                The schedule for mixing the guidance latent. A good starting point is a gentle schedule like `(0.0, 0.3)`, which ramps up the guidance.
            is_panorama (`bool`, *optional*, defaults to False):
                Whether the generation is for a 360-degree panoramic video. Affects tiling logic.
            tile_resolution (`str`, *optional*, defaults to None):
                The resolution of each tile in `WxH` format (e.g., "832x480"). Overrides tile window factors.

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        low_res_video_to_return = None



        if use_coarse_to_fine:
            print("--- Coarse-to-Fine Generation Stage 1: Generating low-resolution video ---")
            low_res_max_area = int(max_area * coarse_max_area_factor)
            low_res_video = self._generate_coarse_video(
                input_prompt=input_prompt,
                img=img,
                max_area=low_res_max_area,
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=coarse_sampling_steps,
                guide_scale=guide_scale,
                n_prompt=n_prompt,
                seed=seed,
                offload_model=offload_model
            )
            low_res_video_to_return = low_res_video

            print("--- Coarse-to-Fine Generation Stage 2: Preparing for high-resolution generation ---")
            h_img_for_size, w_img_for_size = TF.to_tensor(img).shape[1:]
            aspect_ratio_for_size = h_img_for_size / w_img_for_size
            lat_h_hr = round(
                np.sqrt(max_area * aspect_ratio_for_size) / self.vae_stride[1] /
                self.patch_size[1] * self.patch_size[1])
            lat_w_hr = round(
                np.sqrt(max_area / aspect_ratio_for_size) / self.vae_stride[2] /
                self.patch_size[2] * self.patch_size[2])

            print(f"Encoding and upscaling low-resolution video latent to {lat_h_hr}x{lat_w_hr} for guidance...")
            low_res_latent_list = self.vae.encode([low_res_video.to(self.device)])
            if low_res_latent_list:
                low_res_latent = low_res_latent_list[0]
                low_res_latent_f_batch = low_res_latent.permute(1, 0, 2, 3)
                guidance_latent_hr_f_batch = torch.nn.functional.interpolate(
                    low_res_latent_f_batch,
                    size=(lat_h_hr, lat_w_hr),
                    mode='bicubic',
                    align_corners=False
                )
                guidance_latent_hr = guidance_latent_hr_f_batch.permute(1, 0, 2, 3)
            else:
                guidance_latent_hr = None
        else:
            guidance_latent_hr = None

        img_tensor_for_clip_vae = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h_img, w_img = img_tensor_for_clip_vae.shape[1:]
        aspect_ratio = h_img / w_img

        lat_h = round(
            np.sqrt(max_area * aspect_ratio) / self.vae_stride[1] /
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) / self.vae_stride[2] /
            self.patch_size[2] * self.patch_size[2])
        
        h_target_render = lat_h * self.vae_stride[1]
        w_target_render = lat_w * self.vae_stride[2]

        F_model_sched = (F - 1) // self.config.vae_stride[0] + 1
        F_model_raw = F
        
        full_max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        full_max_seq_len = int(math.ceil(full_max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        C_noise = 16
        noise = torch.randn(
            C_noise, F_model_sched,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        msk_frames_model = F_model_sched
        
        C_mask = 4
        msk_cond = torch.ones(1, 1, lat_h, lat_w, device=self.device)
        if F_model_sched > 1:
            msk_zeros = torch.zeros(1, F_model_sched - 1, lat_h, lat_w, device=self.device)
            msk_cond_frames = torch.cat([msk_cond, msk_zeros], dim=1)
        else:
            msk_cond_frames = msk_cond

        _msk_temp = torch.ones(1, F, lat_h, lat_w, device=self.device)
        _msk_temp[:, 1:] = 0
        
        _raw_msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        _raw_msk[:, 1:] = 0
        
        msk_builder = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk_builder[:, 1:] = 0
        msk_final_channels = 4
        msk_y_cond_part = torch.zeros(msk_final_channels, F_model_sched, lat_h, lat_w, device=self.device)
        if F_model_sched > 0:
            msk_y_cond_part[:, 0, :, :] = 1

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model: self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        self.clip.model.to(self.device)
        clip_context = self.clip.visual([img_tensor_for_clip_vae[:, None, :, :]])
        if offload_model: self.clip.model.cpu()

        # The y_condition should always be based on the initial input image to provide a consistent
        # starting point for style, even when using coarse-to-fine guidance for motion.
        vae_input_img_part = torch.nn.functional.interpolate(
            img_tensor_for_clip_vae[None], size=(h_target_render, w_target_render), mode='bicubic'
        ).squeeze(0)
        
        if F > 1:
            vae_input_zeros_part = torch.zeros(vae_input_img_part.shape[0], F - 1, h_target_render, w_target_render,
                                               dtype=vae_input_img_part.dtype, device=self.device)
            vae_input_full = torch.cat([vae_input_img_part.unsqueeze(1), vae_input_zeros_part], dim=1)
        else:
            vae_input_full = vae_input_img_part.unsqueeze(1)

        vae_y_encoded_list = self.vae.encode([vae_input_full.to(self.device)])
        vae_y_encoded = vae_y_encoded_list[0] if vae_y_encoded_list else torch.zeros(C_noise, F_model_sched, lat_h, lat_w, device=self.device)


        y_condition = torch.cat([msk_y_cond_part, vae_y_encoded], dim=0)

        @contextmanager
        def noop_no_sync(): yield
        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(sample_scheduler, device=self.device, sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            if use_coarse_to_fine and guidance_latent_hr is not None:
                timesteps = sample_scheduler.timesteps
                # timesteps = sample_scheduler.timesteps[coarse_sampling_steps:]
                noise = torch.randn_like(guidance_latent_hr)
                # Get the noise at the specific timestep
                # latent_xt = sample_scheduler.add_noise(guidance_latent_hr, noise, timesteps[0])
                K = len(timesteps)
                step_a = 0
                step_b = K - coarse_sampling_steps
                latent_xt = self.re_noise(guidance_latent_hr, noise, step_a, step_b, sample_scheduler)
            else:
                latent_xt = noise.to(self.device)
                timesteps = sample_scheduler.timesteps

            denoised_x0 = None

            if offload_model: torch.cuda.empty_cache()
            self.model.to(self.device)

            if enable_tiling:
                # Tiling logic starts here
                full_latent_h, full_latent_w = lat_h, lat_w
                patch_h_model, patch_w_model = self.model.patch_size[1], self.model.patch_size[2]

                if tile_resolution:
                    tile_w_pixels, tile_h_pixels = map(int, tile_resolution.split('x'))
                    tile_h_lat = (tile_h_pixels // self.vae_stride[1] // patch_h_model) * patch_h_model
                    tile_w_lat = (tile_w_pixels // self.vae_stride[2] // patch_w_model) * patch_w_model
                else:
                    tile_h_lat = int(full_latent_h * tile_window_height_factor)
                    tile_w_lat = int(full_latent_w * tile_window_width_factor)

                stride_h = int(tile_h_lat * tile_stride_factor)
                stride_w = int(tile_w_lat * tile_stride_factor)

                latent_handler = RingLatent2D(latent_xt, wrap=is_panorama)
                noise_pred_aggregator = torch.zeros_like(latent_xt, device=self.device)

            for i_step, t_val in enumerate(tqdm(timesteps)):
                current_timestep_tensor = torch.tensor([t_val], device=self.device)

                if enable_tiling:
                    noise_pred_aggregator.zero_()
                    num_tiles = 0
                    for h_start in range(0, full_latent_h, stride_h):
                        for w_start in range(0, full_latent_w, stride_w):
                            h_end = min(h_start + tile_h_lat, full_latent_h)
                            w_end = min(w_start + tile_w_lat, full_latent_w)

                            tile_latent = latent_handler.get_window_latent(top=h_start, bottom=h_end, left=w_start, right=w_end)

                            model_input_x_list = [tile_latent]
                            model_input_y_list = [y_condition] # This needs to be tiled as well

                            model_kwargs_tile = {
                                'context': context,
                                'clip_fea': clip_context,
                                'seq_len': full_max_seq_len, # This might need adjustment for tiles
                                'y': model_input_y_list,
                            }
                            noise_pred_cond_tile = self.model(model_input_x_list, t=current_timestep_tensor, **model_kwargs_tile)[0]

                            model_kwargs_null_tile = {
                                'context': context_null,
                                'clip_fea': clip_context,
                                'seq_len': full_max_seq_len, # This might need adjustment for tiles
                                'y': model_input_y_list,
                            }
                            noise_pred_uncond_tile = self.model(model_input_x_list, t=current_timestep_tensor, **model_kwargs_null_tile)[0]

                            noise_pred_tile = noise_pred_uncond_tile + guide_scale * (noise_pred_cond_tile - noise_pred_uncond_tile)

                            # Aggregate noise predictions
                            noise_pred_aggregator[:, :, :, h_start:h_end, w_start:w_end] += noise_pred_tile
                            num_tiles += 1

                    noise_pred = noise_pred_aggregator / num_tiles

                else: # Non-tiled generation
                    model_input_x_list = [latent_xt]
                    model_input_y_list = [y_condition]

                    model_kwargs_full = {
                        'context': context,
                        'clip_fea': clip_context,
                        'seq_len': full_max_seq_len,
                        'y': model_input_y_list,
                    }
                    noise_pred_cond = self.model(model_input_x_list, t=current_timestep_tensor, **model_kwargs_full)[0]

                    model_kwargs_null_full = {
                        'context': context_null,
                        'clip_fea': clip_context,
                        'seq_len': full_max_seq_len,
                        'y': model_input_y_list,
                    }
                    noise_pred_uncond = self.model(model_input_x_list, t=current_timestep_tensor, **model_kwargs_null_full)[0]
                    noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

                step_output = sample_scheduler.step(noise_pred, t_val, latent_xt, return_dict=True, generator=seed_g)
                latent_xt = step_output.prev_sample
                if hasattr(step_output, 'pred_original_sample'):
                    denoised_x0 = step_output.pred_original_sample

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                if enable_tiling and tile_resolution:
                    videos = self._decode_tiled(denoised_x0 if denoised_x0 is not None else latent_xt, tile_resolution)
                else:
                    videos = self.vae.decode([denoised_x0 if denoised_x0 is not None else latent_xt])

            del noise, y_condition, context, context_null, clip_context, sample_scheduler, latent_xt

            if offload_model:
                gc.collect()
                torch.cuda.synchronize()
            if dist.is_initialized():
                dist.barrier()

            return (videos[0], low_res_video_to_return) if self.rank == 0 else (None, None)