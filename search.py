# search.py

import faiss
import pickle
import numpy as np
from feature_extractor import FeatureExtractor


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


extractor = FeatureExtractor()

clip_index = faiss.read_index("db/clip.index")
isc_index = faiss.read_index("db/isc.index")

with open("db/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)


def search(query_path, k=5):
    feats = extractor.extract(query_path)

    query_clip = l2_normalize(feats["clip"]).astype("float32").reshape(1, -1)
    query_isc = l2_normalize(feats["isc"]).astype("float32").reshape(1, -1)

    D_clip, I_clip = clip_index.search(query_clip, k)
    D_isc, I_isc = isc_index.search(query_isc, k)

    candidate_idxs = set(I_clip[0].tolist()) | set(I_isc[0].tolist())
    results = []

    for idx in candidate_idxs:
        clip_vec_db = clip_index.reconstruct(int(idx))
        isc_vec_db = isc_index.reconstruct(int(idx))

        clip_score = float(np.dot(query_clip[0], clip_vec_db))
        isc_score = float(np.dot(query_isc[0], isc_vec_db))

        # Decision logic
        if isc_score > 0.9:
            label = "duplicate"
        elif clip_score > 0.8:
            label = "similar"
        else:
            label = "different"

        results.append({
            "file": metadata[idx]["file"],
            "clip_score": clip_score,
            "isc_score": isc_score,
            "label": label
        })

    results.sort(key=lambda r: max(r["isc_score"], r["clip_score"]), reverse=True)
    return results[:k]
