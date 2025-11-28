import torch
from src.model import SimpleCNN


def test_model_binary_output_shape():
    """Default num_classes=2"""
    model = SimpleCNN()
    model.eval()

    x = torch.randn(1, 3, 224, 224)
    out = model(x)

    assert out.shape == (1, 2)


def test_model_three_class_output_shape():
    """Custom 3-class medical setup"""
    model = SimpleCNN(num_classes=3)
    model.eval()

    x = torch.randn(4, 3, 224, 224)
    out = model(x)

    assert out.shape == (4, 3)


def test_model_no_nan_outputs():
    model = SimpleCNN(num_classes=3)
    model.eval()

    x = torch.randn(1, 3, 224, 224)
    out = model(x)

    assert torch.isfinite(out).all()

