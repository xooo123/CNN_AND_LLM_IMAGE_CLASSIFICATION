import sys
import os
from fastapi.testclient import TestClient

# --- MOCK MODEL LOADING BEFORE IMPORTING app ---
import torch
from types import SimpleNamespace


def mock_load_checkpoint(path, device):
    class DummyModel:
        def __call__(self, x):
            return torch.randn(1, 2)

        def eval(self):
            return self

        def to(self, device):
            return self

    return DummyModel(), ["Normal", "COVID"]


# Inject mock BEFORE importing model_server
import model_server
model_server.load_checkpoint = mock_load_checkpoint
model_server.model, model_server.CLASS_NAMES = mock_load_checkpoint(None, None)

from model_server import app

client = TestClient(app)


def test_health_endpoint_works():
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "device" in data
    assert "classes" in data

