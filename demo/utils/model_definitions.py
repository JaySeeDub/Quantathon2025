# All imports
from .imports import *
from .Preprocessing import *
from .Helper import *
warnings.filterwarnings('ignore')

# Binary DN without Quantum Layer
class BinaryDNN_classical(nn.Module):
    def __init__(self):
        super().__init__()

        # Encodes features from dataset
        self.feature_encoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
            # nn.Linear(128, 1024),
            # nn.LeakyReLU(0.2),
            # nn.Dropout(0.2)
        )

        # Classifies based on encoded features
        self.classifier = nn.Sequential(
            # nn.Linear(1024, 128),
            # nn.LeakyReLU(0.2),
            # nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        feats_encoded = self.feature_encoder(features)
        class_probs = self.classifier(feats_encoded)

        return class_probs  # Shape: (batch_size, 1)