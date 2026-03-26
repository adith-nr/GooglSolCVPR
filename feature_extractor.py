# feature_extractor.py

import torch
import clip
import numpy as np
from PIL import Image
from isc_feature_extractor import create_model

class FeatureExtractor:
    def __init__(self):
        # CLIP
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32")

        # ISC
        self.isc_model, self.isc_preprocess = create_model(
            weight_name='isc_ft_v107', device='cpu'
        )

    def extract_clip(self, image):
        image = self.clip_preprocess(image).unsqueeze(0)
        with torch.no_grad():
            embedding = self.clip_model.encode_image(image)
        return embedding.squeeze().numpy()

    def extract_isc(self, image):
        image = self.isc_preprocess(image).unsqueeze(0)
        with torch.no_grad():
            embedding = self.isc_model(image)
        return embedding.squeeze().numpy()

    def extract(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return {
            "clip": self.extract_clip(image),
            "isc": self.extract_isc(image)
        }
