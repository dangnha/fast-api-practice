import io

import pytest
from PIL import Image

from app.ml import MnistClassifier


@pytest.fixture(scope="module")
def model():
    return MnistClassifier("models/mnist_cnn.pt")


def _make_png() -> bytes:
    image = Image.new("L", (28, 28), color=0)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_model_is_deterministic(model):
    content = _make_png()
    first = model.predict(content)
    second = model.predict(content)
    assert first == second
    assert 0.0 <= first.confidence <= 1.0


def test_model_rejects_empty_input(model):
    with pytest.raises(ValueError, match="Empty input"):
        model.predict(b"")
