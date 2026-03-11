import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src import models
from src.methods import vton
from src.utils import (
    IImage,
    resize,
    poisson_blend,
    warping_cloth,
    get_densepose,
    DensePose,
)
from src.flux.FluxVtonPipeline import FluxVtonPipeline
from diffusers.pipelines.flux.pipeline_flux_fill import FluxFillPipeline

logging.disable(logging.INFO)

negative_prompt = (
    "text, bad anatomy, bad proportions, blurry, cropped, deformed, disfigured, "
    "duplicate, error, extra limbs, gross proportions, jpeg artifacts, long neck, "
    "low quality, lowres, malformed, morbid, mutated, mutilated, out of frame, ugly, worst quality"
)
positive_prompt = "Full HD, 4K, high quality, high resolution"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image-path", type=Path, required=True, help="Person image path")
    parser.add_argument("--cloth-path", type=Path, required=True, help="Cloth image path")
    parser.add_argument("--condition-dir", type=Path, required=True, help="Condition directory")
    parser.add_argument("--output-path", type=Path, required=True, help="Final output image path")

    parser.add_argument("--model-id", type=str, default="sd2_inp", help='One of [sd15_inp, sd2_inp, flux_inp]')
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--W", type=int, default=384)
    parser.add_argument("--device", type=str, default="cuda:0")

    return parser.parse_args()


def require_file(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {name} -> {path}")
    return path


def optional_file(path: Path):
    return path if path.exists() else None


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_prompt(json_path: Path) -> str:
    data = load_json(json_path)
    for key in ["cloth_clip_interrogate", "prompt", "caption", "text"]:
        if key in data:
            return data[key]
    raise KeyError(f"Cannot find prompt text in {json_path}")


def load_pose_keypoints(json_path: Path):
    data = load_json(json_path)
    people = data.get("people", [])
    if len(people) < 1:
        print(f"[WARN] No pose_keypoints_2d found in {json_path}")
        return None

    pose = people[0].get("pose_keypoints_2d", None)
    if pose is None:
        print(f"[WARN] pose_keypoints_2d missing in {json_path}")
        return None

    pose = np.array(pose, dtype=np.float32).reshape((-1, 3))
    return pose


def load_uv(path: Path):
    if path is None:
        return None

    uv = get_densepose(str(path))
    uv = np.transpose(uv, (1, 2, 0))
    if np.all(uv == 0):
        return None
    return uv


def load_meta(meta_path: Path):
    meta = load_json(meta_path)

    skin_color = torch.tensor(meta.get("skin_color", [255, 255, 255]), dtype=torch.float32)
    c_type = str(meta.get("c_type", "0"))
    sub_type = int(meta.get("sub_type", -1))

    return {
        "skin_color": skin_color,
        "c_type": c_type,
        "sub_type": sub_type,
    }


def resolve_condition_paths(condition_dir: Path):
    paths = {
        "in_mask": require_file(condition_dir / "agnostic_mask.png", "agnostic_mask.png"),
        "out_mask": require_file(condition_dir / "cloth_mask.png", "cloth_mask.png"),
        "parse_mask": require_file(condition_dir / "parse_mask.png", "parse_mask.png"),
        "densepose_mask": require_file(condition_dir / "densepose_mask.png", "densepose_mask.png"),
        "pseudo": require_file(condition_dir / "pseudo.jpg", "pseudo.jpg"),
        "pseudo_mask": require_file(condition_dir / "pseudo_mask.png", "pseudo_mask.png"),
        "human_pose": require_file(condition_dir / "human_pose.json", "human_pose.json"),
        "cloth_pose": require_file(condition_dir / "cloth_pose.json", "cloth_pose.json"),
        "human_parsing": require_file(condition_dir / "human_parsing.png", "human_parsing.png"),
        "cloth_parsing": require_file(condition_dir / "cloth_parsing.png", "cloth_parsing.png"),
        "cloth_prompt": require_file(condition_dir / "cloth_prompt.json", "cloth_prompt.json"),
        "meta": require_file(condition_dir / "meta.json", "meta.json"),
        "image_uv": optional_file(condition_dir / "image_uv.pkl"),
        "cloth_uv": optional_file(condition_dir / "cloth_uv.pkl"),
    }
    return paths


def get_inpainting_function(
    model,
    negative_prompt: str = "",
    positive_prompt: str = "",
    num_steps: int = 50,
    guidance_scale: float = 7.5,
):
    def run(
        image: Image.Image,
        cloth: Image.Image,
        warped_cloth,
        in_mask: Image.Image,
        out_mask: Image.Image,
        parse_mask: Image.Image,
        densepose_mask: Image.Image,
        skin_color: torch.Tensor,
        warped_dp,
        cprompt: str,
        seed: int = 1,
    ):
        if isinstance(model, (FluxVtonPipeline, FluxFillPipeline)):
            painted_image = model(
                prompt_image=cprompt,
                prompt_cloth=cprompt,
                image=image,
                image_mask=in_mask,
                cloth=cloth,
                cloth_mask=out_mask,
                warp_cloth=warped_cloth,
                densepose_mask=densepose_mask,
                warped_dp=warped_dp,
                parse_mask=parse_mask,
                skin_color=skin_color,
                height=image.size[1],
                width=image.size[0],
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                seed=seed
            ).images
        else:
            painted_image = vton.run(
                ddim=model,
                cprompt=cprompt,
                image=IImage(image),
                cloth=cloth,
                warped_cloth=warped_cloth,
                in_mask=IImage(in_mask),
                out_mask=IImage(out_mask),
                parse_mask=IImage(parse_mask),
                densepose_mask=IImage(densepose_mask),
                skin_color=skin_color,
                warped_dp=warped_dp,
                seed=seed,
                negative_prompt=negative_prompt,
                positive_prompt=positive_prompt,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
            ).pil()

        w, h = image.size
        inpainted_image = Image.fromarray(np.array(painted_image[1])[:h, -w:])
        outpainted_image = Image.fromarray(np.array(painted_image[0])[:h, -w:])
        return outpainted_image, inpainted_image

    return run


def build_warped_cloth(
    W,
    H,
    c_type,
    sub_type,
    resized_cloth,
    resized_in_mask,
    ri_cloth,
    ri_out_mask,
    human_pose_path,
    cloth_pose_path,
    human_parsing_path,
    cloth_parsing_path,
):
    human_pose_data = load_pose_keypoints(human_pose_path)
    cloth_pose_data = load_pose_keypoints(cloth_pose_path)

    if human_pose_data is None or cloth_pose_data is None:
        return None, None

    ori_parsing = Image.open(human_parsing_path).convert("RGB")
    out_parsing = Image.open(cloth_parsing_path).convert("RGB")

    resized_ori_parsing = resize(ori_parsing, (W, H))
    resized_out_parsing = resize(out_parsing, (W, H))

    out_mask_array = np.array(ri_out_mask)
    in_mask_array = np.array(resized_in_mask)
    ori_parsing_array = np.array(resized_ori_parsing)
    out_parsing_array = np.array(resized_out_parsing)
    cloth_array = np.array(ri_cloth)

    cloth_array = cloth_array / 127.5 - 1
    ori_parsing_array = ori_parsing_array / 127.5 - 1
    out_parsing_array = out_parsing_array / 127.5 - 1
    mask_array = out_mask_array / 127.5 - 1

    cloth_array[out_mask_array == 0] = 0
    out_parsing_array[out_mask_array == 0] = 0
    mask_array[out_mask_array == 0] = 0

    w, h = resized_cloth.size

    if c_type == "2":
        upper_dress = cloth_array.copy()
        lower_dress = cloth_array.copy()
        upper_mask = mask_array.copy()
        lower_mask = mask_array.copy()

        warped_upper_mask = warping_cloth(
            upper_mask, out_parsing_array, ori_parsing_array,
            cloth_pose_data, human_pose_data, w, h,
            c_type="0", sub_type=sub_type
        )
        warped_lower_mask = warping_cloth(
            lower_mask, out_parsing_array, ori_parsing_array,
            cloth_pose_data, human_pose_data, w, h,
            c_type="1", sub_type=1
        )
        warped_upper = warping_cloth(
            upper_dress, out_parsing_array, ori_parsing_array,
            cloth_pose_data, human_pose_data, w, h,
            c_type="0", sub_type=sub_type
        )
        warped_lower = warping_cloth(
            lower_dress, out_parsing_array, ori_parsing_array,
            cloth_pose_data, human_pose_data, w, h,
            c_type="1", sub_type=1
        )

        warped_lower[warped_upper != 0] = 0
        warped_lower_mask[warped_upper_mask != 0] = 0
        warped_cloth = warped_upper + warped_lower
        warped_mask = warped_upper_mask + warped_lower_mask
    else:
        warped_mask = warping_cloth(
            mask_array, out_parsing_array, ori_parsing_array,
            cloth_pose_data, human_pose_data, w, h,
            c_type=c_type, sub_type=sub_type
        )
        warped_cloth = warping_cloth(
            cloth_array, out_parsing_array, ori_parsing_array,
            cloth_pose_data, human_pose_data, w, h,
            c_type=c_type, sub_type=sub_type
        )

    warped_cloth[in_mask_array == 0] = 0
    warped_mask[in_mask_array == 0] = 0

    warped_cloth = ((warped_cloth + 1) * 127.5).clip(0, 255).astype(np.uint8)
    warped_mask = ((warped_mask + 1) * 127.5).clip(0, 255).astype(np.uint8)

    return warped_cloth, warped_mask


def build_warped_densepose(
    H,
    W,
    pseudo_mask,
    resized_in_mask,
    warped_mask,
    image_uv_path,
    cloth_uv_path,
    device,
):
    p_uv = load_uv(image_uv_path)
    c_uv = load_uv(cloth_uv_path)

    resized_warped_dp = None

    if p_uv is not None and c_uv is not None:
        dp = DensePose((H, W), oob_ocluded=True, naive_warp=False)
        grid = dp.get_grid_warp(c_uv, p_uv)

        grid_tensor = torch.from_numpy(grid).float().unsqueeze(0).to(device)
        pseudo_mask_t = IImage(pseudo_mask).torch(0).to(device)
        mask_t = IImage(resized_in_mask).torch(0).to(device)

        warped_dp = F.grid_sample(pseudo_mask_t, grid_tensor, align_corners=True)
        resized_warped_dp = warped_dp * mask_t

    if warped_mask is not None:
        if resized_warped_dp is None:
            resized_warped_dp = warped_mask
        else:
            if isinstance(resized_warped_dp, torch.Tensor):
                warped_np = resized_warped_dp.detach().cpu().numpy().astype(np.float32)
                if warped_np.ndim == 4 and warped_np.shape[0] == 1:
                    warped_np = warped_np[0]
                if warped_np.ndim == 3 and warped_np.shape[0] == 3:
                    warped_np = np.transpose(warped_np, (1, 2, 0))
            else:
                warped_np = np.array(resized_warped_dp, dtype=np.float32)

            if warped_np.max() > 1.0:
                warped_np /= 255.0

            warped_mask_np = warped_mask.astype(np.float32)
            if warped_mask_np.max() > 1.0:
                warped_mask_np /= 255.0

            warped_mask_np[warped_mask_np < 0.6] = 0
            merge_dp = np.clip(warped_np + warped_mask_np, 0.0, 1.0)
            resized_warped_dp = (merge_dp * 255).clip(0, 255).astype(np.uint8)

    return resized_warped_dp


def main():
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    cond = resolve_condition_paths(args.condition_dir)
    meta = load_meta(cond["meta"])

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = models.load_inpainting_model(args.model_id, device=str(device), cache=True)

    run_inpainting = get_inpainting_function(
        model=model,
        guidance_scale=args.guidance_scale,
        num_steps=args.num_steps,
        negative_prompt=negative_prompt,
        positive_prompt=positive_prompt,
    )

    image = Image.open(args.image_path).convert("RGB")
    cloth = Image.open(args.cloth_path).convert("RGB")
    in_mask = Image.open(cond["in_mask"]).convert("RGB")
    out_mask = Image.open(cond["out_mask"]).convert("RGB")
    parse_mask = Image.open(cond["parse_mask"]).convert("RGB")
    densepose_mask = Image.open(cond["densepose_mask"]).convert("RGB")
    pseudo = Image.open(cond["pseudo"]).convert("RGB")
    pseudo_mask = Image.open(cond["pseudo_mask"]).convert("RGB")
    cprompt = load_prompt(cond["cloth_prompt"])

    resized_image = resize(image, (args.W, args.H))
    resized_cloth = resize(cloth, (args.W, args.H))
    resized_in_mask = resize(in_mask, (args.W, args.H), resample=Image.NEAREST)
    resized_out_mask = resize(out_mask, (args.W, args.H), resample=Image.NEAREST)
    resized_parse_mask = resize(parse_mask, (args.W, args.H), resample=Image.NEAREST)
    resized_densepose_mask = resize(densepose_mask, (args.W, args.H), resample=Image.NEAREST)

    ri_cloth = resize(pseudo, (args.W, args.H))
    ri_out_mask = resize(pseudo_mask, (args.W, args.H), resample=Image.NEAREST)

    warped_cloth, warped_mask = build_warped_cloth(
        W=args.W,
        H=args.H,
        c_type=meta["c_type"],
        sub_type=meta["sub_type"],
        resized_cloth=resized_cloth,
        resized_in_mask=resized_in_mask,
        ri_cloth=ri_cloth,
        ri_out_mask=ri_out_mask,
        human_pose_path=cond["human_pose"],
        cloth_pose_path=cond["cloth_pose"],
        human_parsing_path=cond["human_parsing"],
        cloth_parsing_path=cond["cloth_parsing"],
    )

    out_mask_array = np.array(resized_out_mask)
    out_mask_array = 255 - out_mask_array
    resized_out_mask = Image.fromarray(out_mask_array)

    resized_warped_dp = build_warped_densepose(
        H=args.H,
        W=args.W,
        pseudo_mask=pseudo_mask,
        resized_in_mask=resized_in_mask,
        warped_mask=warped_mask,
        image_uv_path=cond["image_uv"],
        cloth_uv_path=cond["cloth_uv"],
        device=device,
    )

    for idx in range(args.num_samples):
        seed = args.seed + idx * 1000

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()

        _, inpainted_image = run_inpainting(
            resized_image,
            resized_cloth,
            warped_cloth,
            resized_in_mask,
            resized_out_mask,
            resized_parse_mask,
            resized_densepose_mask,
            meta["skin_color"],
            resized_warped_dp,
            cprompt,
            seed=seed
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.time()
        print(f"Inference time: {end - start:.3f}s")

        blended = poisson_blend(
            orig_img=IImage(resized_image).data[0],
            fake_img=IImage(inpainted_image).data[0],
            mask=IImage(resized_in_mask).alpha().data[0],
        )
        blended = Image.fromarray(IImage(blended).data[0])

        if args.num_samples == 1:
            save_path = args.output_path
        else:
            save_path = args.output_path.with_name(
                f"{args.output_path.stem}_seed{seed}{args.output_path.suffix}"
            )

        blended.save(save_path)
        print(f"saved to: {save_path}")


if __name__ == "__main__":
    main()
