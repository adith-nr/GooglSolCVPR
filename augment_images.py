#!/usr/bin/env python3

import argparse
import os
import random
from typing import List

from PIL import Image, ImageFilter


def list_images(root: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if os.path.splitext(name.lower())[1] in exts:
                files.append(os.path.join(dirpath, name))
    files.sort()
    return files


def random_crop(img: Image.Image, min_scale: float, max_scale: float) -> Image.Image:
    w, h = img.size
    scale = random.uniform(min_scale, max_scale)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    if new_w == w and new_h == h:
        return img

    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)
    return img.crop((left, top, left + new_w, top + new_h))


def random_resize(img: Image.Image, min_scale: float, max_scale: float) -> Image.Image:
    w, h = img.size
    scale = random.uniform(min_scale, max_scale)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def apply_random_ops(
    img: Image.Image,
    blur_prob: float,
    crop_prob: float,
    resize_prob: float,
) -> Image.Image:
    out = img

    if random.random() < crop_prob:
        out = random_crop(out, 0.6, 1.0)

    if random.random() < resize_prob:
        out = random_resize(out, 0.5, 1.5)

    if random.random() < blur_prob:
        radius = random.uniform(0.5, 3.0)
        out = out.filter(ImageFilter.GaussianBlur(radius=radius))

    return out


def save_variants(
    image_path: str,
    out_dir: str,
    num_variants: int,
    blur_prob: float,
    crop_prob: float,
    resize_prob: float,
) -> None:
    base = os.path.splitext(os.path.basename(image_path))[0]
    img = Image.open(image_path).convert("RGB")

    for i in range(num_variants):
        variant = apply_random_ops(img, blur_prob, crop_prob, resize_prob)
        out_path = os.path.join(out_dir, f"{base}_aug_{i+1}.jpg")
        variant.save(out_path, quality=90, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create random blur/crop/resize variants of images."
    )
    parser.add_argument("--input-dir", default="images")
    parser.add_argument("--output-dir", default="images")
    parser.add_argument("--num-variants", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--blur-prob", type=float, default=0.6)
    parser.add_argument("--crop-prob", type=float, default=0.7)
    parser.add_argument("--resize-prob", type=float, default=0.7)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    images = list_images(args.input_dir)
    if not images:
        raise SystemExit(f"No images found in {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    for path in images:
        save_variants(
            path,
            args.output_dir,
            args.num_variants,
            args.blur_prob,
            args.crop_prob,
            args.resize_prob,
        )


if __name__ == "__main__":
    main()
