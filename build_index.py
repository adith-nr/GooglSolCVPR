# build_index.py

import os
import pickle
import argparse
from typing import List, Dict

import faiss
import numpy as np
from PIL import Image


def list_images(root: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if os.path.splitext(name.lower())[1] in exts:
                files.append(os.path.join(dirpath, name))
    files.sort()
    return files


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def simple_features(image_path: str) -> Dict[str, np.ndarray]:
    # Lightweight fallback: resize and flatten RGB pixels.
    image = Image.open(image_path).convert("RGB")
    image = image.resize((32, 32))
    vec = np.asarray(image, dtype=np.float32).reshape(-1) / 255.0
    return {"clip": vec, "isc": vec}


def build_indexes(
    image_dir: str,
    out_dir: str,
    simple: bool = False,
) -> None:
    if simple:
        extractor = None
    else:
        from feature_extractor import FeatureExtractor
        extractor = FeatureExtractor()
    image_paths = list_images(image_dir)

    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    clip_vecs = []
    isc_vecs = []
    metadata: List[Dict[str, str]] = []

    for path in image_paths:
        feats = simple_features(path) if simple else extractor.extract(path)
        clip_vecs.append(l2_normalize(feats["clip"]).astype("float32"))
        isc_vecs.append(l2_normalize(feats["isc"]).astype("float32"))
        metadata.append({"file": path})

    clip_matrix = np.vstack(clip_vecs)
    isc_matrix = np.vstack(isc_vecs)

    clip_index = faiss.IndexFlatIP(clip_matrix.shape[1])
    isc_index = faiss.IndexFlatIP(isc_matrix.shape[1])

    clip_index.add(clip_matrix)
    isc_index.add(isc_matrix)

    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(clip_index, os.path.join(out_dir, "clip.index"))
    faiss.write_index(isc_index, os.path.join(out_dir, "isc.index"))

    with open(os.path.join(out_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--out-dir", default="db")
    parser.add_argument("--simple", action="store_true", help="Use lightweight image features (no CLIP/ISC).")
    args = parser.parse_args()

    build_indexes(args.image_dir, args.out_dir, simple=args.simple)


if __name__ == "__main__":
    main()
