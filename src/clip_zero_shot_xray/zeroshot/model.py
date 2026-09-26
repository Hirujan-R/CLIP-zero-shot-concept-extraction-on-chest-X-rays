"""Vision-language model loading and embedding helpers."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import open_clip
import torch
from PIL import Image

DEFAULT_MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def pick_device(prefer: str = "auto") -> str:
    """Resolve a torch device, defaulting to Apple MPS, then CUDA, then CPU."""
    if prefer != "auto":
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ZeroShotModel:
    """Thin wrapper around an open_clip model exposing image/text embeddings."""

    def __init__(self, name: str = DEFAULT_MODEL, device: str = "auto") -> None:
        self.name = name
        self.device = pick_device(device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(name)
        self.tokenizer = open_clip.get_tokenizer(name)
        self.model = self.model.to(self.device).eval()
        self.logit_scale = self.model.logit_scale.exp().detach().cpu()

    @torch.no_grad()
    def encode_images(self, images: Image.Image | list[Image.Image]) -> np.ndarray:
        """Embed one or more PIL images, returning L2-normalised rows."""
        if isinstance(images, Image.Image):
            images = [images]
        batch = torch.stack(
            [self.preprocess(image.convert("RGB")) for image in images]
        ).to(self.device)
        embeddings = self.model.encode_image(batch)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy()

    @torch.no_grad()
    def encode_texts(self, texts: str | list[str]) -> np.ndarray:
        """Embed one or more prompts, returning L2-normalised rows."""
        if isinstance(texts, str):
            texts = [texts]
        tokens = self.tokenizer(list(texts)).to(self.device)
        embeddings = self.model.encode_text(tokens)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy()


@lru_cache(maxsize=2)
def get_model(name: str = DEFAULT_MODEL, device: str = "auto") -> ZeroShotModel:
    """Return a cached model instance so weights load only once per process."""
    return ZeroShotModel(name=name, device=device)
