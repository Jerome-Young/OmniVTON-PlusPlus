import cv2
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from tqdm import tqdm

from src.utils.iimage import IImage
from src.smplfusion import share, router, attentionpatch, transformerpatch

verbose = False
_CODEBOOK_SIZE = 64


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


def init_guidance() -> None:
    router.attention_forward = attentionpatch.default.forward_and_save
    router.basic_transformer_forward = transformerpatch.default.forward


def generate_codebook_element(t: int, k: int, img_size, K: int):
    seed = int(t * K + k)
    generator = torch.Generator(device='cuda')
    generator.manual_seed(seed)
    return torch.randn(img_size, generator=generator, device='cuda')


def run(
    ddim,
    cprompt,
    image,
    cloth,
    in_mask,
    out_mask,
    warped_cloth=None,
    parse_mask=None,
    densepose_mask=None,
    skin_color=None,
    warped_dp=None,
    querys=None,
    keys=None,
    seed=0,
    negative_prompt='',
    positive_prompt='',
    num_steps=50,
    guidance_scale=7.5,
):
    image = image.padx(64)
    in_mask = in_mask.dilate(1).alpha().padx(64)
    out_mask = out_mask.dilate(1).alpha().padx(64)

    full_cprompt = f'{cprompt}, {positive_prompt}' if positive_prompt else cprompt

    init_guidance()
    share.set_mask(out_mask)

    dt = 1000 // num_steps
    device = torch.device('cuda')

    cc = ddim.encoder.encode(full_cprompt)
    uc = ddim.encoder.encode(negative_prompt)
    context = torch.cat([uc, uc, cc, cc], dim=0)

    in_mask_tensor = in_mask.torch(0).to(device)
    out_mask_tensor = 1 - out_mask.torch(0).to(device)

    cloth = IImage(cloth).padx(64)
    out_unet_condition = ddim.get_inpainting_condition(cloth, out_mask)

    if warped_cloth is not None:
        warped_cloth = IImage(warped_cloth).padx(64)
        in_unet_condition = ddim.get_inpainting_condition(image, in_mask, warped_cloth)
    else:
        in_unet_condition = ddim.get_inpainting_condition(image, in_mask)

    dtype = out_unet_condition.dtype
    unet_condition = torch.cat([out_unet_condition, in_unet_condition], dim=0)

    seed_everything(seed)
    z0 = torch.randn((1, 4, *unet_condition.shape[2:]), device=device, dtype=dtype)

    warped_dp_tensor = None
    if warped_dp is not None:
        warped_dp_tensor = IImage(warped_dp).padx(64).torch().to(device=device, dtype=dtype)

    combine_img = replace_masked_region(
        image.torch().to(device),
        cloth.torch().to(device),
        parse_mask.torch(0).to(device),
        out_mask_tensor,
        densepose_mask.torch(0).to(device),
        skin_color=skin_color,
        warped_dp=warped_dp_tensor,
    )

    p_z0 = (ddim.vae.encode(combine_img.to(dtype)).mean * ddim.config.scale_factor).to(dtype)

    zt = z0.repeat(2, 1, 1, 1)
    ddim.unet.requires_grad_(False)

    iterator = tqdm(range(999, 0, -dt)) if verbose else range(999, 0, -dt)
    autocast_enabled = device.type == 'cuda'

    for timestep in share.DDIMIterator(iterator):
        zt = zt.detach()
        zt.requires_grad_(False)

        model_input = torch.cat([zt, unet_condition], dim=1)
        timesteps = torch.full((4,), timestep, device=device, dtype=torch.long)

        with torch.autocast(device_type='cuda', enabled=autocast_enabled):
            eps_uncond, eps_cond = ddim.unet(
                torch.cat([model_input, model_input], dim=0).to(dtype),
                timesteps=timesteps,
                context=context,
                in_mask=in_mask_tensor,
                out_mask=out_mask_tensor,
                querys=querys,
                keys=keys,
                time=timestep,
            ).detach().chunk(2)

        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        z0 = (zt - share.schedule.sqrt_one_minus_alphas[timestep] * eps) / share.schedule.sqrt_alphas[timestep]
        _, in_z0 = z0.chunk(2)

        codebook = torch.stack(
            [generate_codebook_element(timestep, k, in_z0.shape[1:], _CODEBOOK_SIZE).to(dtype) for k in range(_CODEBOOK_SIZE)],
            dim=0)

        in_pose, _ = decompose_pose_texture(in_z0, n_components=3)
        residual = (p_z0 - in_pose).to(dtype=torch.float32)

        sims = torch.einsum('kuvw,buvw->kb', codebook.to(torch.float32), residual)
        noise = codebook[torch.argmax(sims, dim=0)]

        sigma = torch.sqrt(
            (1 - share.schedule.alphas[timestep - dt])
            / (1 - share.schedule.alphas[timestep])
            * (1 - share.schedule.alphas[timestep] / share.schedule.alphas[timestep - dt])
        )

        zt = (
            share.schedule.sqrt_alphas[timestep - dt] * z0
            + torch.sqrt(1 - share.schedule.alphas[timestep - dt] - sigma ** 2) * eps
            + sigma * noise
        )

    with torch.no_grad():
        output_image = IImage(ddim.vae.decode(z0 / ddim.config.scale_factor))

    return output_image