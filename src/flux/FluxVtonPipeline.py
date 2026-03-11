# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import cv2

import numpy as np
import torch
from traitlets import Bool
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from pytorch_lightning import seed_everything
from torch import einsum
import torch.nn.functional as F
from src.utils.iimage import IImage

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from .transformer_flux import FluxTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from .flux_attn_processor import FluxVtonAttnProcessor
from diffusers.models.attention_processor import FluxAttnProcessor2_0
from diffusers.schedulers.scheduling_ddim_inverse import DDIMInverseScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import SchedulerOutput
from sklearn.decomposition import PCA

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import FluxFillPipeline
        >>> from diffusers.utils import load_image

        >>> image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/cup.png")
        >>> mask = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/cup_mask.png")

        >>> pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16)
        >>> pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU

        >>> image = pipe(
        ...     prompt_image="a white paper cup",
        ...     image=image,
        ...     image_mask=mask,
        ...     height=1632,
        ...     width=1232,
        ...     guidance_scale=30,
        ...     num_inference_steps=50,
        ...     max_sequence_length=512,
        ...     generator=torch.Generator("cpu").manual_seed(0),
        ... ).images[0]
        >>> image.save("flux_fill.png")
        ```
"""


# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
        scheduler,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
        encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def _resize_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if mask.shape[-2:] == size:
        return mask
    return F.interpolate(mask, size=size, mode='nearest')


def _masked_average_color(image: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    b, c, _, _ = image.shape
    image_flat = image.flatten(2)
    mask_flat = mask.flatten(1)
    weighted_sum = torch.einsum('bcn,bn->bc', image_flat, mask_flat)
    total_weight = mask_flat.sum(dim=-1, keepdim=True)
    avg_color = weighted_sum / (total_weight + eps)
    return avg_color.view(b, c, 1, 1)


def _expand_color(color, ref: torch.Tensor) -> torch.Tensor | None:
    if color is None:
        return None
    b, c, h, w = ref.shape
    color = color.to(device=ref.device, dtype=ref.dtype)
    if color.ndim == 1:
        color = color.view(1, c, 1, 1)
    return color.expand(b, c, h, w)


def replace_masked_region(
    p_img: torch.Tensor,
    c_img: torch.Tensor,
    p_mask: torch.Tensor,
    c_mask: torch.Tensor,
    d_mask: torch.Tensor,
    skin_color=None,
    warped_dp=None,
    eps: float = 1e-6,
) -> torch.Tensor:
    _, _, h, w = p_img.shape
    p_mask = _resize_mask(p_mask, (h, w))
    c_mask = _resize_mask(c_mask, (h, w))
    d_mask = _resize_mask(d_mask, (h, w))

    avg_color = _masked_average_color(c_img, c_mask, eps=eps)

    base_img = p_img * (1 - p_mask)
    base_np = IImage(base_img).data[0]
    mask_np = IImage(p_mask, 0, 1).data[0][:, :, 0]
    inpainted_np = cv2.inpaint(base_np, mask_np, 3, cv2.INPAINT_NS)
    combine_img = IImage(inpainted_np).padx(64).torch().to(device=p_img.device, dtype=p_img.dtype)

    skin_mask = p_mask * d_mask
    expanded_skin = _expand_color(skin_color, combine_img)
    if expanded_skin is not None:
        combine_img = combine_img * (1 - skin_mask) + expanded_skin * skin_mask
    else:
        combine_img = combine_img * (1 - skin_mask)

    if warped_dp is not None:
        warped_dp = warped_dp.to(device=combine_img.device, dtype=combine_img.dtype)
        if warped_dp.min() < -0.5:
            warped_dp = (warped_dp + 1) / 2
        combine_img = combine_img * (1 - warped_dp) + avg_color.to(combine_img.dtype) * warped_dp

    return combine_img


# def decompose_pose_texture(z0: torch.Tensor, n_components: int = 8) -> tuple:
#     z = z0.squeeze(0).to(torch.float32)
#     z_pose = torch.zeros_like(z)
#     z_texture = torch.zeros_like(z)
#     for c in range(z.shape[0]):
#         channel_data = z[c].cpu().numpy()
#
#         X = channel_data
#         pca = PCA(n_components=n_components)
#         X_pca = pca.fit_transform(X)
#
#         X_recon_pose = pca.inverse_transform(X_pca)
#         X_texture = channel_data - X_recon_pose
#         z_pose[c] = torch.tensor(X_recon_pose).to(z0.dtype)
#         z_texture[c] = torch.tensor(X_texture).to(z0.dtype)
#
#     return z_pose.unsqueeze(0), z_texture.unsqueeze(0)


def decompose_pose_texture(z0: torch.Tensor, n_components: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    z = z0.squeeze(0).to(dtype=torch.float32)
    pose = torch.empty_like(z)
    texture = torch.empty_like(z)

    q = min(n_components, z.shape[-2], z.shape[-1])
    for c in range(z.shape[0]):
        x = z[c]  # [H, W]
        u, s, v = torch.pca_lowrank(x, q=q, center=False)
        pose_c = (u * s.unsqueeze(0)) @ v.transpose(0, 1)
        pose[c] = pose_c
        texture[c] = x - pose_c

    return pose.unsqueeze(0).to(dtype=z0.dtype), texture.unsqueeze(0).to(dtype=z0.dtype)


def generate_codebook_element(t: int, k: int, img_size, K: int):
    seed = int(t * K + k)
    generator = torch.Generator(device='cuda')
    generator.manual_seed(seed)
    return torch.randn(img_size, generator=generator, device='cuda')


def step(model, model_output, timestep, sample, image_pose, height, width, variance_noise=None, return_dict=True):
    scheduler = model.scheduler
    if scheduler.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    if scheduler.step_index is None:
        scheduler._init_step_index(timestep)

    # Improve numerical stability for small number of steps
    lower_order_final = (scheduler.step_index == len(scheduler.timesteps) - 1) and (
        scheduler.config.euler_at_final
        or (scheduler.config.lower_order_final and len(scheduler.timesteps) < 15)
        or scheduler.config.final_sigmas_type == "zero"
    )
    lower_order_second = (
        (scheduler.step_index == len(scheduler.timesteps) - 2) and scheduler.config.lower_order_final and len(scheduler.timesteps) < 15
    )

    model_output = scheduler.convert_model_output(model_output, sample=sample)
    for i in range(scheduler.config.solver_order - 1):
        scheduler.model_outputs[i] = scheduler.model_outputs[i + 1]
    scheduler.model_outputs[-1] = model_output

    # Upcast to avoid precision issues when computing prev_sample
    sample = sample.to(torch.float32)
    if scheduler.config.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"] and variance_noise is None:
        K = 64
        codebook = torch.stack(
            [generate_codebook_element(timestep, k, image_pose.shape[1:], K).to(model_output.dtype) for k in range(K)],
            dim=0)
        x0 = model_output
        out_x0, in_x0 = x0.chunk(2)
        in_x0 = model._unpack_latents(in_x0, height, width, model.vae_scale_factor)
        in_pose, in_texture = decompose_pose_texture(in_x0, n_components=3)
        height = 2 * (int(height) // (model.vae_scale_factor * 2))
        width = 2 * (int(width) // (model.vae_scale_factor * 2))
        residual = image_pose - in_pose
        sims = einsum('k u w v, b u w v->k b', codebook, residual.to(model_output.dtype))
        idxs = torch.argmax(sims, dim=0)
        noise = codebook[idxs]
        noise = model._pack_latents(noise, noise.shape[0], model.vae.config.latent_channels, height, width)
    elif scheduler.config.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"]:
        noise = variance_noise.to(device=model_output.device, dtype=torch.float32)
    else:
        noise = None

    if scheduler.config.solver_order == 1 or scheduler.lower_order_nums < 1 or lower_order_final:
        prev_sample = scheduler.dpm_solver_first_order_update(model_output, sample=sample, noise=noise)
    elif scheduler.config.solver_order == 2 or scheduler.lower_order_nums < 2 or lower_order_second:
        prev_sample = scheduler.multistep_dpm_solver_second_order_update(scheduler.model_outputs, sample=sample, noise=noise)
    else:
        prev_sample = scheduler.multistep_dpm_solver_third_order_update(scheduler.model_outputs, sample=sample, noise=noise)

    if scheduler.lower_order_nums < scheduler.config.solver_order:
        scheduler.lower_order_nums += 1

    # Cast sample back to expected dtype
    prev_sample = prev_sample.to(model_output.dtype)

    # upon completion increase step index by one
    scheduler._step_index += 1

    if not return_dict:
        return (prev_sample,)

    return SchedulerOutput(prev_sample)


class FluxVtonPipeline(
    DiffusionPipeline,
    FluxLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin,
):
    r"""
    The Flux Fill pipeline for image inpainting/outpainting.

    Reference: https://blackforestlabs.ai/flux-1-tools/

    Args:
        transformer ([`FluxTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`T5TokenizerFast`):
            Second Tokenizer of class
            [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        ddim_inversion_scheduler: DDIMInverseScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
    ):
        super().__init__()
        scheduler = DPMSolverMultistepScheduler(prediction_type='flow_prediction', flow_shift=3.0, use_flow_sigmas=True,
                                                algorithm_type='sde-dpmsolver++')

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
            ddim_inversion_scheduler=ddim_inversion_scheduler,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        # Flux latents are turned into 2x2 patches and packed. This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2,
            vae_latent_channels=self.vae.config.latent_channels,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = 128

    def _set_attn_processor(self, AttnProcessor=FluxAttnProcessor2_0):
        attn_processor_dict = {}
        for k in self.transformer.attn_processors.keys():
            attn_processor_dict[k] = AttnProcessor()

        self.transformer.set_attn_processor(attn_processor_dict)

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._get_t5_prompt_embeds
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer_2)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]

        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._get_clip_prompt_embeds
    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    def prepare_mask_latents(
        self,
        mask,
        masked_image,
        batch_size,
        num_channels_latents,
        num_images_per_prompt,
        height,
        width,
        dtype,
        device,
        generator,
    ):
        # 1. calculate the height and width of the latents
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        # 2. encode the masked image
        if masked_image.shape[1] == num_channels_latents:
            masked_image_latents = masked_image
        else:
            masked_image_latents = retrieve_latents(self.vae.encode(masked_image), generator=generator)

        masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)

        # 3. duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        batch_size = batch_size * num_images_per_prompt
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

        # 4. pack the masked_image_latents
        # batch_size, num_channels_latents, height, width -> batch_size, height//2 * width//2 , num_channels_latents*4
        masked_image_latents = self._pack_latents(
            masked_image_latents,
            batch_size,
            num_channels_latents,
            height,
            width,
        )

        # 5.resize mask to latents shape we we concatenate the mask to the latents
        mask = mask[:, 0, :, :]  # batch_size, 8 * height, 8 * width (mask has not been 8x compressed)
        mask = mask.view(
            batch_size, height, self.vae_scale_factor, width, self.vae_scale_factor
        )  # batch_size, height, 8, width, 8
        mask = mask.permute(0, 2, 4, 1, 3)  # batch_size, 8, 8, height, width
        mask = mask.reshape(
            batch_size, self.vae_scale_factor * self.vae_scale_factor, height, width
        )  # batch_size, 8*8, height, width

        # 6. pack the mask:
        # batch_size, 64, height, width -> batch_size, height//2 * width//2 , 64*2*2
        mask = self._pack_latents(
            mask,
            batch_size,
            self.vae_scale_factor * self.vae_scale_factor,
            height,
            width,
        )
        mask = mask.to(device=device, dtype=dtype)

        return mask, masked_image_latents

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        lora_scale: Optional[float] = None,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # We only use the pooled prompt output from the CLIPTextModel
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        if self.text_encoder is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
        image=None,
        mask_image=None,
        masked_image_latents=None,
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. Dimensions will be resized accordingly"
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")

        if image is not None and masked_image_latents is not None:
            raise ValueError(
                "Please provide either  `image` or `masked_image_latents`, `masked_image_latents` should not be passed."
            )

        if image is not None and mask_image is None:
            raise ValueError("Please provide `mask_image` when passing `image`.")

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._prepare_latent_image_ids
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._pack_latents
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._unpack_latents
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i: i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        return image_latents

    @torch.no_grad()
    def obtain_pose(
            self,
            image,
            num_channels_latents,
            dtype=None,
            height=None,
            width=None,
            resize_mode="default",
            crops_coords=None,
    ):

        image = image.to(dtype)

        x0 = self.vae.encode(image.to(self.device)).latent_dist.sample()
        x0 = (x0 - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        x0 = x0.to(dtype)

        # gt_pose, gt_texture = decompose_pose_texture(x0, n_components=3)

        return x0

    def prepare_latents(
                self,
                batch_size,
                num_channels_latents,
                height,
                width,
                dtype,
                device,
                generator,
        ):
            # VAE applies 8x compression on images but we must also account for packing which requires
            # latent height and width to be divisible by 2.
            height = 2 * (int(height) // (self.vae_scale_factor * 2))
            width = 2 * (int(width) // (self.vae_scale_factor * 2))

            shape = (batch_size, num_channels_latents, height, width)
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
            ref_image_ids = self._prepare_latent_image_ids(batch_size, height, width // 2, device, dtype)

            return latents, latent_image_ids, ref_image_ids

    def prepare_image_latents(
                self,
                image,
                batch_size,
                num_channels_latents,
                height,
                width,
                dtype,
                device,
                generator,
                latents=None,
        ):
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            # VAE applies 8x compression on images but we must also account for packing which requires
            # latent height and width to be divisible by 2.
            height = 2 * (int(height) // (self.vae_scale_factor * 2))
            width = 2 * (int(width) // (self.vae_scale_factor * 2))
            shape = (batch_size, num_channels_latents, height, width)
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

            if latents is not None:
                return latents.to(device=device, dtype=dtype), latent_image_ids

            image = image.to(device=device, dtype=dtype)
            image_latents = self._encode_vae_image(image=image, generator=generator)
            if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
                # expand init_latents for batch_size
                additional_image_per_prompt = batch_size // image_latents.shape[0]
                image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
            elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
                )
            else:
                image_latents = torch.cat([image_latents], dim=0)

            latents = image_latents
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
            return latents, latent_image_ids

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt


    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
            self,
            prompt_image: Union[str, List[str]] = None,
            prompt_image_2: Optional[Union[str, List[str]]] = None,
            prompt_cloth: Union[str, List[str]] = None,
            prompt_cloth_2: Optional[Union[str, List[str]]] = None,
            image: Optional[torch.FloatTensor] = None,
            image_mask: Optional[torch.FloatTensor] = None,
            cloth: Optional[torch.FloatTensor] = None,
            cloth_mask: Optional[torch.FloatTensor] = None,
            warp_cloth: Optional[torch.FloatTensor] = None,
            densepose_mask: Optional[torch.FloatTensor] = None,
            warped_dp: Optional[torch.FloatTensor] = None,
            parse_mask: Optional[torch.FloatTensor] = None,
            skin_color: Optional[torch.FloatTensor] = None,
            masked_image_latents: Optional[torch.FloatTensor] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            sigmas: Optional[List[float]] = None,
            guidance_scale: float = 30.0,
            num_images_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            max_sequence_length: int = 512,
            seed: Optional[int] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt_image (`str` or `List[str]`, *optional*):
                The prompt_image or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_image_2 (`str` or `List[str]`, *optional*):
                The prompt_image or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt_image` is
                will be used instead
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
                numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
                or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
                list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)`.
            image_mask (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to mask `image`. White pixels in the mask
                are repainted while black pixels are preserved. If `image_mask` is a PIL image, it is converted to a
                single channel (luminance) before use. If it's a numpy array or pytorch tensor, it should contain one
                color channel (L) instead of 3, so the expected shape for pytorch tensor would be `(B, 1, H, W)`, `(B,
                H, W)`, `(1, H, W)`, `(H, W)`. And for numpy array would be for `(B, H, W, 1)`, `(B, H, W)`, `(H, W,
                1)`, or `(H, W)`.
            mask_image_latent (`torch.Tensor`, `List[torch.Tensor]`):
                `Tensor` representing an image batch to mask `image` generated by VAE. If not provided, the mask
                latents tensor will ge generated by `image_mask`.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt_image`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt_image.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt_image weighting. If not
                provided, text embeddings will be generated from `prompt_image` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt_image weighting.
                If not provided, pooled text embeddings will be generated from `prompt_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt_image`.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """

        seed_everything(seed)
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt_image,
            prompt_image_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
            image=image,
            mask_image=image_mask,
            masked_image_latents=masked_image_latents,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt_image is not None and isinstance(prompt_image, str):
            batch_size = 1
        elif prompt_image is not None and isinstance(prompt_image, list):
            batch_size = len(prompt_image)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Prepare prompt_image embeddings
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt_cloth,
            prompt_2=prompt_cloth_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt * 2,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 4. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        latents, latent_image_ids, ref_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt * 2,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )
        device = self._execution_device
        dtype = prompt_embeds.dtype

        out_mask_tensor = IImage(cloth_mask).dilate(1).alpha().padx(64).torch(0).to(device=device, dtype=dtype)
        parse_mask_tensor = IImage(parse_mask).dilate(1).alpha().padx(64).torch(0).to(device=device, dtype=dtype)

        if densepose_mask is not None:
            densepose_mask_tensor = IImage(densepose_mask).dilate(1).alpha().padx(64).torch(0).to(device=device, dtype=dtype)
        else:
            densepose_mask_tensor = torch.zeros_like(parse_mask_tensor)
        if warped_dp is not None:
            warped_dp = IImage(warped_dp).padx(64).torch().to(device=device, dtype=dtype)
        if warp_cloth is not None:
            warp_cloth = IImage(warp_cloth).padx(64).torch().to(device=device, dtype=dtype)
        if skin_color is not None:
            skin_color = skin_color.to(device=device, dtype=dtype)

        combine_img = replace_masked_region(
            IImage(image).torch().to(device=device, dtype=dtype),
            IImage(cloth).torch().to(device=device, dtype=dtype),
            parse_mask_tensor,
            out_mask_tensor,
            densepose_mask_tensor,
            skin_color=skin_color,
            warped_dp=warped_dp,
        )

        image_pose = self.obtain_pose(
            combine_img, num_channels_latents, height=height, width=width, dtype=prompt_embeds.dtype
        )
        # 5. Prepare mask and masked image latents
        if masked_image_latents is not None:
            masked_image_latents = masked_image_latents.to(latents.device)
        else:
            image = self.image_processor.preprocess(image, height=height, width=width)
            cloth = self.image_processor.preprocess(cloth, height=height, width=width)
            image_mask = self.mask_processor.preprocess(image_mask, height=height, width=width)
            cloth_mask = self.mask_processor.preprocess(cloth_mask, height=height, width=width)
            parse_mask = self.mask_processor.preprocess(parse_mask, height=height, width=width)
            if warp_cloth is None:
                masked_image = image * (1 - image_mask)
            else:
                masked_image = image * (1 - image_mask) + image_mask * warp_cloth.to(image.device)

            masked_cloth = cloth * cloth_mask
            ref_mask = 1 - cloth_mask

            masked_image = masked_image.to(device=device, dtype=prompt_embeds.dtype)
            masked_cloth = masked_cloth.to(device=device, dtype=prompt_embeds.dtype)
            mask, masked_image_latents = self.prepare_mask_latents(
                torch.cat([ref_mask, image_mask]),
                torch.cat([masked_cloth, masked_image]),
                batch_size * 2,
                num_channels_latents,
                num_images_per_prompt,
                image.shape[-2],
                image.shape[-1],
                prompt_embeds.dtype,
                device,
                generator,
            )
            masked_image_latents = torch.cat((masked_image_latents, mask), dim=-1)

        # 6. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # self._set_attn_processor(FluxAttnProcessor2_0)
        self._set_attn_processor(FluxVtonAttnProcessor)
        self._joint_attention_kwargs = self._joint_attention_kwargs or {}
        self._joint_attention_kwargs['in_mask'] = image_mask.to(latents.device)
        self._joint_attention_kwargs['out_mask'] = cloth_mask.to(latents.device)
        self._joint_attention_kwargs['height'] = height
        self._joint_attention_kwargs['width'] = width

        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                input_image_latents = masked_image_latents

                noise_pred = self.transformer(
                    hidden_states=torch.cat((latents, input_image_latents), dim=2),
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    ref_img_ids=ref_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = step(self, noise_pred, t, latents, image_pose, height, width, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        # 8. Post-process the image
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)

