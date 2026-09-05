"""Tiny CNN for handwritten digit classification (MNIST).

Kept intentionally small so inference runs comfortably on a laptop CPU. The
rest of the application only depends on the ``MnistClassifier.predict``
interface, so swapping this for a larger or GPU-hosted model later does not
require touching the HTTP layer.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch import nn


@dataclass(frozen=True)
class ModelPrediction:
    label: str
    confidence: float


class MnistCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


class MnistClassifier:
    """Loads a trained :class:`MnistCNN` and exposes a bytes -> prediction API."""

    def __init__(self, model_path: str) -> None:
        self.model = MnistCNN()
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def predict(self, content: bytes) -> ModelPrediction:
        if not content:
            raise ValueError("Empty input")

        tensor = self._preprocess(content)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, index = probs.max(dim=1)
        return ModelPrediction(
            label=str(index.item()),
            confidence=round(float(confidence.item()), 6),
        )

    @staticmethod
    def _preprocess(content: bytes) -> torch.Tensor:
        image = Image.open(io.BytesIO(content)).convert("L")
        image = image.resize((28, 28), Image.Resampling.BILINEAR)

        # MNIST is white-on-black; many uploads are black-on-white. Normalizing
        # per-image by its own range keeps either convention working.
        pixels = np.frombuffer(image.tobytes(), dtype=np.uint8).copy()
        tensor = torch.from_numpy(pixels).float().view(28, 28)
        lo, hi = tensor.min(), tensor.max()
        if hi > lo:
            tensor = (tensor - lo) / (hi - lo)
        return tensor.view(1, 1, 28, 28)
