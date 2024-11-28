# predict_trump_with_nn.py

import torch
import pickle
from pathlib import Path
import warnings
from jass.game.const import *

# Suppress the specific FutureWarning for `weights_only=False` in torch.load
warnings.filterwarnings("ignore", category=FutureWarning,
                        message="You are using `torch.load` with `weights_only=False`")


class TrumpPredictor:
    def __init__(self):
        # Load the entire model directly
        self.model = torch.load('models/trump_predictor.pth')
        self.model.eval()  # Set the model to evaluation mode

        # Load the label encoder
        with open('encoder/label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)

    def predict(self, input_features):
        """Predict the trump value based on input features."""
        # Convert input features to a tensor
        input_tensor = torch.tensor(input_features.values, dtype=torch.float32)

        with torch.no_grad():
            # Perform prediction
            outputs = self.model(input_tensor)
            _, predicted_idx = torch.max(outputs, 1)

        prediction_int = [predicted_idx.item()][0]

        return prediction_int  # Convert to integer using helper function if needed
