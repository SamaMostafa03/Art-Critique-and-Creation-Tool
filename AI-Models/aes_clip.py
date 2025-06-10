import torch
import torch.nn as nn
import clip
from PIL import Image
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score , mean_absolute_error
import time
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
class ResidualBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(features, features * 4),
            nn.GELU(),
            nn.Linear(features * 4, features),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return x + self.block(x)

class AesCLIP_reg(nn.Module):
    def __init__(self, clip_name):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load CLIP model and get feature size
        self.aesclip, self.clip_size = self.select_clip(clip_name)

        #  Define the MLP for aesthetic score regression
        self.mlp = nn.Sequential(
            nn.Linear(self.clip_size, 512),
            nn.BatchNorm1d(512),  # Helps stabilize training
            nn.GELU(),
            ResidualBlock(512),
            nn.Dropout(0.3),  # Reduce dropout
            nn.Linear(512, 1)
        )

        #  Initialize model weights
        self.init_weights()

    def select_clip(self, clip_name):
        """Load the correct CLIP model based on the given clip_name."""
        model, _ = clip.load(clip_name, device=self.device)
        clip_size = model.visual.output_dim  # Extract the feature size

        return model, clip_size

    def init_weights(self):
        """Initialize weights with Kaiming Normal for relu-based networks."""
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')  # Use 'relu' instead of 'gelu'
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 6.0)  # Set bias initialization to dataset mean

    def forward(self, x):
        """Extract features using CLIP and predict aesthetic score."""
        with torch.no_grad():  # CLIP feature extraction
            features = self.aesclip.encode_image(x).float()

        return self.mlp(features).squeeze(dim=-1)
