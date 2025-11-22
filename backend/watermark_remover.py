#!/usr/bin/env python
"""
Automatic watermark remover.

If you pass multiple images that share the same watermark, the detector learns
what stays constant across the set (edges with low variance) and builds a mask
before inpainting. Single-image runs fall back to a heuristic detector that
hunts for watermark-like strokes.
"""
from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def normalize_to_unit(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    min_v = float(arr.min())
    max_v = float(arr.max())
    if max_v - min_v < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - min_v) / (max_v - min_v + 1e-6)


def collect_paths(patterns: Sequence[str]) -> List[Path]:
    seen = set()
    paths: List[Path] = []
    for pattern in patterns:
        matches = glob(pattern)
        if not matches and Path(pattern).is_file():
            matches = [pattern]
        for match in matches:
            p = Path(match).expanduser().resolve()
            if not p.is_file():
                continue
            if p.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            if p in seen:
                continue
            seen.add(p)
            paths.append(p)
    return paths


def load_images(paths: Sequence[Path]) -> List[np.ndarray]:
    images: List[np.ndarray] = []
    for path in paths:
        with Image.open(path) as img:
            rgb = np.array(img.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        images.append(bgr)
    return images


def resize_for_stack(images: Sequence[np.ndarray]) -> Tuple[List[np.ndarray], Tuple[int, int]]:
    min_w = min(img.shape[1] for img in images)
    min_h = min(img.shape[0] for img in images)
    if len(images) == 1:
        return list(images), (min_w, min_h)
    resized = [
        cv2.resize(img, (min_w, min_h), interpolation=cv2.INTER_AREA)
        if img.shape[0] != min_h or img.shape[1] != min_w
        else img
        for img in images
    ]
    return resized, (min_w, min_h)


def refine_mask(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, open_kernel, iterations=1)
    mask_u8 = cv2.dilate(mask_u8, open_kernel, iterations=1)
    return mask_u8


def otsu_mask(score: np.ndarray) -> np.ndarray:
    score_norm = normalize_to_unit(score)
    if score_norm.max() <= 0:
        return np.zeros_like(score_norm, dtype=np.uint8)
    score_u8 = np.clip(score_norm * 255.0, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(score_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def compute_multi_image_mask(images: Sequence[np.ndarray]) -> Tuple[np.ndarray, Tuple[int, int]]:
    resized, size = resize_for_stack(images)
    stack = np.stack(resized).astype(np.float32) / 255.0
    gray_stack = stack.mean(axis=3)
    var = gray_stack.var(axis=0)
    mean_gray = gray_stack.mean(axis=0)
    grad_x = cv2.Sobel(mean_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(mean_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    grad = normalize_to_unit(grad)
    inv_var = normalize_to_unit(1.0 / (var + 1e-6))
    score = grad * inv_var  # high when edges stay constant while backgrounds vary
    score = cv2.GaussianBlur(score, (5, 5), 0)
    mask = otsu_mask(score)
    mask = refine_mask(mask)
    return mask, size


def compute_single_image_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    grad = normalize_to_unit(grad)
    local_mean = cv2.blur(gray.astype(np.float32) / 255.0, (35, 35))
    contrast = np.abs(gray.astype(np.float32) / 255.0 - local_mean)
    contrast = normalize_to_unit(contrast)
    score = grad * (1.0 - 0.5 * contrast)
    score = cv2.GaussianBlur(score, (5, 5), 0)
    mask = otsu_mask(score)
    mask = refine_mask(mask)
    return mask


def inpaint_image(image: np.ndarray, mask: np.ndarray, radius: float) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    return cv2.inpaint(image, mask_u8, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)


def save_image(image: np.ndarray, path: Path) -> None:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(path)


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(mask.astype(np.uint8)).save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automatically remove watermarks from one or more images.")
    parser.add_argument("inputs", nargs="+", help="Image paths or glob patterns.")
    parser.add_argument("-o", "--output-dir", default="backend/outputs", help="Where to save cleaned images.")
    parser.add_argument("--save-masks", action="store_true", help="Save the detected masks for inspection.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing outputs.")
    parser.add_argument("--inpaint-radius", type=float, default=3.0, help="Inpaint radius passed to OpenCV.")
    parser.add_argument("--verbose", action="store_true", help="Print extra info about the detector.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = collect_paths(args.inputs)
    if not paths:
        raise SystemExit("No valid image files found.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = load_images(paths)

    if len(images) > 1:
        mask, base_size = compute_multi_image_mask(images)
        detector_note = f"multi-image detector at size {base_size[0]}x{base_size[1]}"
    else:
        mask = compute_single_image_mask(images[0])
        detector_note = "single-image heuristic detector"

    if args.verbose:
        mask_ratio = float(np.count_nonzero(mask)) / float(mask.size)
        print(f"Using {detector_note}. Mask covers {mask_ratio:.3%} of pixels.")

    for path, img in tqdm(zip(paths, images), total=len(images), desc="Cleaning"):
        resized_mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        cleaned = inpaint_image(img, resized_mask, radius=args.inpaint_radius)

        out_name = f"{path.stem}_cleaned{path.suffix}"
        out_path = output_dir / out_name
        if out_path.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite existing file: {out_path}")
        save_image(cleaned, out_path)

        if args.save_masks:
            mask_path = output_dir / f"{path.stem}_mask.png"
            save_mask(resized_mask, mask_path)

    if args.verbose:
        print(f"Saved results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
