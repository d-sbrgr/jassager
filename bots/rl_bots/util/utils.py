# utils.py

import torch

from bots.rl_bots.util.jassnet import JassNet

def save_model(model, filepath="rl_model.pth"):
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(model_class, filepath="rl_model.pth"):
    model = model_class(input_dim = 481, action_dim = 36)  # Instantiate the model
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {filepath}")
    return model
