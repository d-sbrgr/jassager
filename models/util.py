import torch
import torch.nn as nn
from pathlib import Path
from typing import Type

def get_model_storage_path() -> Path:
    return Path(__file__).parent / "parameters"

def save_model(model, name: str, version: int):
    location = get_model_storage_path() / f"{name}_{version}.pth"
    torch.save(model.state_dict(), location)
    print(f"Model saved at {location}")

def load_model(model_class: Type[nn.Module], model_name, version):
    location = get_model_storage_path() / f"{model_name}_{version}.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_class()
    model.load_state_dict(torch.load(location, map_location=device, weights_only=True))
    model.eval()
    return model