import torch
import torch.nn as nn
from pathlib import Path

def get_model_storage_path() -> Path:
    return Path(__file__).parent / "parameters"

def save_model(model, name: str, version: int):
    location = get_model_storage_path() / f"{name}_{version}.pth"
    torch.save(model.state_dict(), location)
    print(f"Model saved at {location}")

def load_model(model_class: nn.Module, name: str, version: int) -> nn.Module:
    location = get_model_storage_path() / f"{name}_{version}.pth"
    model = model_class()
    model.load_state_dict(torch.load(location, weights_only=True))
    model.eval()
    return model