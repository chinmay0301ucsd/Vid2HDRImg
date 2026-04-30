# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# SVD pipeline conditioned on:
#   1. CLIP image embedding of the first frame (encoder_hidden_states, cross-attention)
#   2. VAE latent of the first frame concatenated with noisy latents (channel-wise)
# No ControlNet.

import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from diffusers.models import AutoencoderKLTemporalDecoder
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers import UNetSpatioTemporalConditionModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _get_tile_views(
    lat_h: int,
    lat_w: int,
    tile_h: int = 64,
    tile_w: int = 64,
    overlap_ratio: float = 0.5,
):
    """
    Generate overlapping tile coordinates in latent space for MultiDiffusion inference.

    Replicates get_views() from dataset_crop_MStack_valid but with configurable square
    tile dimensions suited for the HDR pipeline (512×512 px → 64×64 latent).

    Args:
        lat_h, lat_w:   Full latent spatial dimensions (height // 8, width // 8).
        tile_h, tile_w: Tile size in latent space (pixel_tile_size // 8).
        overlap_ratio:  Fractional overlap between adjacent tiles [0, 1).

    Returns:
        List of (w_start, h_start, w_end, h_end) tuples in latent-space coordinates.
    """
    stride_h = max(1, int(tile_h * (1 - overlap_ratio)))
    stride_w = max(1, int(tile_w * (1 - overlap_ratio)))

    n_h = (lat_h - tile_h) // stride_h + 1
    if (lat_h - tile_h) % stride_h != 0:
        n_h += 1
    if lat_h <= tile_h:
        n_h = 1

    n_w = (lat_w - tile_w) // stride_w + 1
    if (lat_w - tile_w) % stride_w != 0:
        n_w += 1
    if lat_w <= tile_w:
        n_w = 1

    views = []
    for i in range(n_h * n_w):
        h_start = int((i // n_w) * stride_h)
        h_end = h_start + tile_h
        if h_end > lat_h:
            h_end = lat_h
            h_start = h_end - tile_h
        w_start = int((i % n_w) * stride_w)
        w_end = w_start + tile_w
        if w_end > lat_w:
            w_end = lat_w
            w_start = w_end - tile_w
        views.append((w_start, h_start, w_end, h_end))

    return views


def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom sigmas."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@dataclass
class StableVideoDiffusionPipelineOutput(BaseOutput):
    r"""
    Output class for the first-frame-conditioned SVD pipeline.

    Args:
        frames (`List[List[PIL.Image.Image]]` or `np.ndarray` or `torch.Tensor`):
            Generated video frames.
    """
    frames: Union[List[List[PIL.Image.Image]], np.ndarray, torch.Tensor]


class StableVideoDiffusionPipelineHDR(DiffusionPipeline):
    r"""
    SVD pipeline conditioned on:
      1. CLIP image embedding of the first frame via cross-attention (encoder_hidden_states).
      2. VAE latent of the first frame concatenated channel-wise with noisy latents.

    No ControlNet is used.

    Args:
        vae (`AutoencoderKLTemporalDecoder`): Temporal VAE for encoding/decoding.
        image_encoder (`CLIPVisionModelWithProjection`): CLIP vision encoder.
        unet (`UNetSpatioTemporalConditionModel`): SVD denoising UNet.
        scheduler (`EulerDiscreteScheduler`): Noise scheduler.
        feature_extractor (`CLIPImageProcessor`): Preprocessor for CLIP inputs.
    """

    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        vae: AutoencoderKLTemporalDecoder,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNetSpatioTemporalConditionModel,
        scheduler: EulerDiscreteScheduler,
        feature_extractor: CLIPImageProcessor,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.video_processor = VideoProcessor(do_resize=True, vae_scale_factor=self.vae_scale_factor)

    def _encode_image(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.Tensor],
        device: Union[str, torch.device],
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
        clip_image: Optional[Union[PIL.Image.Image, List[PIL.Image.Image]]] = None,
    ):
        """Encode the first frame through CLIP to get encoder_hidden_states [B, 1, D].

        For CFG: returns [zeros; clip_embeds] so uncond is always zeros.

        clip_image: if provided, use this PIL image for CLIP encoding instead of
        image (useful when image is a high-precision float tensor for VAE but CLIP
        needs standard uint8 PIL input with proper feature_extractor preprocessing).
        """
        dtype = next(self.image_encoder.parameters()).dtype

        clip_input = clip_image if clip_image is not None else image
        if not isinstance(clip_input, torch.Tensor):
            image = self.feature_extractor(images=clip_input, return_tensors="pt").pixel_values
        else:
            image = clip_input

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds   # (B, D)
        image_embeddings = image_embeddings.unsqueeze(1)             # (B, 1, D)

        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)
            # uncond first, cond second (matches image_latents ordering)
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

        return image_embeddings

    def _encode_vae_image(
        self,
        image: torch.Tensor,
        device: Union[str, torch.device],
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
        clip_only_cfg: bool = False,
    ):
        image = image.to(device=device, dtype=self.vae.dtype)
        image_latents = self.vae.encode(image).latent_dist.mode()

        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)

        if do_classifier_free_guidance:
            # clip_only_cfg: keep the raw-input latent in the uncond pass — only CLIP is zeroed.
            # Standard CFG: zero both CLIP and VAE latent for uncond.
            negative_image_latents = image_latents if clip_only_cfg else torch.zeros_like(image_latents)
            image_latents = torch.cat([negative_image_latents, image_latents])

        return image_latents

    def _get_add_time_ids(
        self,
        fps: int,
        motion_bucket_id: int,
        noise_aug_strength: float,
        dtype: torch.dtype,
        batch_size: int,
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, "
                f"but a vector of {passed_add_embed_dim} was created."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)

        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids])

        return add_time_ids

    def decode_latents(self, latents: torch.Tensor, num_frames: int, decode_chunk_size: int = 14):
        latents = latents.flatten(0, 1)
        latents = 1 / self.vae.config.scaling_factor * latents
        latents = latents.to(dtype=self.vae.dtype)

        forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
        accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())

        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                decode_kwargs["num_frames"] = num_frames_in
            frame = self.vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)

        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)
        frames = frames.float()
        return frames

    def check_inputs(self, image, height, width):
        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(
                f"`image` must be torch.Tensor, PIL.Image.Image, or list, got {type(image)}"
            )
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` must be divisible by 8, got {height} and {width}.")

    def prepare_latents(
        self,
        batch_size: int,
        num_frames: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: Union[str, torch.device],
        generator: torch.Generator,
        latents: Optional[torch.Tensor] = None,
    ):
        shape = (
            batch_size,
            num_frames,
            num_channels_latents // 2,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective "
                f"batch size of {batch_size}."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        if isinstance(self.guidance_scale, (int, float)):
            return self.guidance_scale > 1
        return self.guidance_scale.max() > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    def __call__(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.Tensor],
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        sigmas: Optional[List[float]] = None,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        clip_image: Optional[Union[PIL.Image.Image, List[PIL.Image.Image]]] = None,
        conditioning_frame_idx: Optional[int] = None,
        clip_only_cfg: bool = False,
        overlap_ratio: Optional[float] = None,
        tile_size: int = 512,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
    ):
        r"""
        Generate a video from the first frame.

        Args:
            image: First frame (PIL Image, list of PIL Images, or tensor [B, 3, H, W]).
            height: Output height in pixels (must be divisible by 8).
            width: Output width in pixels (must be divisible by 8).
            num_frames: Number of frames to generate.
            num_inference_steps: Number of denoising steps.
            min_guidance_scale / max_guidance_scale: CFG scale range across frames.
            fps: FPS conditioning (SVD uses fps-1 internally).
            motion_bucket_id: Motion conditioning bucket.
            noise_aug_strength: Noise added to first frame before VAE encoding.
            conditioning_frame_idx: If None (default), the conditioning latent is repeated
                across all frames (original behavior). If an int, the latent is placed only
                at that temporal position and all other positions are zeroed. Use with models
                trained with --ev_cond_mode.
            decode_chunk_size: Frames decoded at once (reduce if OOM).
            num_videos_per_prompt: Number of videos per input.
            generator: RNG for reproducibility.
            latents: Optional pre-computed noisy latents.
            output_type: "pil", "np", "pt", or "latent".
            return_dict: Return StableVideoDiffusionPipelineOutput if True.
        """
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        self.check_inputs(image, height, width)

        # ── Tiling setup (MultiDiffusion) ────────────────────────────────────────
        use_tiling = overlap_ratio is not None
        if use_tiling:
            lat_h = height // self.vae_scale_factor
            lat_w = width // self.vae_scale_factor
            tile_lat = tile_size // self.vae_scale_factor
            views = _get_tile_views(lat_h, lat_w, tile_lat, tile_lat, overlap_ratio)
            logger.debug(
                f"MultiDiffusion tiling: {len(views)} tiles ({tile_size}×{tile_size} px / "
                f"{tile_lat}×{tile_lat} latent) over {height}×{width} image, "
                f"overlap_ratio={overlap_ratio}"
            )

        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]

        device = self._execution_device
        self._guidance_scale = max_guidance_scale

        # SVD is conditioned on fps - 1
        fps = fps - 1

        # Encode first frame through CLIP (original image, before noise augmentation)
        encoder_hidden_states = self._encode_image(
            image,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            clip_image=clip_image,
        )

        # Preprocess first frame and encode through VAE (with noise augmentation)
        image_preprocessed = self.video_processor.preprocess(image, height=height, width=width).to(device)
        noise = randn_tensor(
            image_preprocessed.shape, generator=generator, device=device, dtype=image_preprocessed.dtype
        )
        image_preprocessed = image_preprocessed + noise_aug_strength * noise

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        if use_tiling:
            # Pre-encode per-tile VAE latents while VAE is at the correct precision.
            # Mirrors the ControlNet pipeline's per-tile VAE encode inside the loop,
            # but pre-computed here for efficiency (conditioning image is static).
            tile_image_latents = []
            for w_start, h_start, w_end, h_end in views:
                h_px, w_px = int(h_start) * 8, int(w_start) * 8
                tile_img = image_preprocessed[:, :, h_px:h_px + tile_size, w_px:w_px + tile_size]
                tile_enc = self._encode_vae_image(
                    tile_img,
                    device=device,
                    num_videos_per_prompt=num_videos_per_prompt,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    clip_only_cfg=clip_only_cfg,
                ).unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
                tile_image_latents.append(tile_enc)
            latents_dtype = tile_image_latents[0].dtype
            image_latents = None   # not used in tiling mode
        else:
            image_latents = self._encode_vae_image(
                image_preprocessed,
                device=device,
                num_videos_per_prompt=num_videos_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                clip_only_cfg=clip_only_cfg,
            )
            latents_dtype = image_latents.dtype

        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # [2B, num_frames, 4, H', W'] — conditioning latent placed across frames
        if not use_tiling:
            image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
            if conditioning_frame_idx is not None:
                # EV-conditioned mode: non-zero only at the specified frame position.
                # The uncond half (first B rows) is already zeros so masking is safe.
                frame_mask = torch.zeros(
                    image_latents.shape[0], num_frames, 1, 1, 1,
                    device=image_latents.device, dtype=image_latents.dtype,
                )
                frame_mask[:, conditioning_frame_idx] = 1.0
                image_latents = image_latents * frame_mask

        # Added time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            latents_dtype,
            batch_size,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        # Timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, None, sigmas
        )

        # Prepare noisy latents
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            latents_dtype,
            device,
            generator,
            latents,
        )

        # Per-frame guidance scale ramp
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents_dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)
        self._guidance_scale = guidance_scale

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if use_tiling:
                    # ── MultiDiffusion: accumulate noise predictions across tiles ────────
                    # Matches the ControlNet pipeline approach: for each tile, run the UNet
                    # on the latent crop + per-tile VAE-encoded conditioning image, then
                    # average noise predictions in overlapping regions (uniform weight = 1).
                    value = torch.zeros_like(latents)   # (B, T, C, H', W')
                    count = torch.zeros_like(latents)

                    for (w_start, h_start, w_end, h_end), img_lat_tile in zip(views, tile_image_latents):
                        w_start, h_start = int(w_start), int(h_start)
                        w_end, h_end = int(w_end), int(h_end)

                        lat_tile = latents[:, :, :, h_start:h_end, w_start:w_end]
                        if self.do_classifier_free_guidance:
                            lat_tile_in = torch.cat([lat_tile] * 2)
                        else:
                            lat_tile_in = lat_tile

                        lat_tile_in = self.scheduler.scale_model_input(lat_tile_in, t)
                        lat_tile_in = torch.cat([lat_tile_in, img_lat_tile], dim=2)

                        noise_tile = self.unet(
                            lat_tile_in,
                            t,
                            encoder_hidden_states=encoder_hidden_states,
                            added_time_ids=added_time_ids,
                            return_dict=False,
                        )[0]

                        if self.do_classifier_free_guidance:
                            noise_uncond, noise_cond = noise_tile.chunk(2)
                            # guidance_scale is (B, T, 1, 1, 1) — broadcasts over spatial dims
                            noise_tile = noise_uncond + self.guidance_scale * (noise_cond - noise_uncond)

                        value[:, :, :, h_start:h_end, w_start:w_end] += noise_tile
                        count[:, :, :, h_start:h_end, w_start:w_end] += 1

                    noise_pred = value / count.clamp(min=1)
                else:
                    # ── Standard single-pass inference ────────────────────────────────────
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    # Concatenate first-frame latent with noisy latents along channel dim
                    latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=encoder_hidden_states,
                        added_time_ids=added_time_ids,
                        return_dict=False,
                    )[0]

                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = self.video_processor.postprocess_video(video=frames, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)
