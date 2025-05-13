# model.py
# Loads the pretrained CVT backbone and replaces its classifier for binary classification

import torch
import torch.nn as nn
from transformers import CvtForImageClassification

class CustomClassifier(nn.Module):
    """
    A custom classifier head to replace the default CVT classifier.
    Architecture:
        - FC(384 ➝ 256) + Mish + BatchNorm + Dropout(0.5)
        - FC(256 ➝ 128) + Mish + BatchNorm + Dropout(0.3)
        - FC(128 ➝ 2) for binary classification (Real vs. AI Generated)
    """
    def __init__(self):
        super(CustomClassifier, self).__init__()

        self.fc1 = nn.Linear(384, 256)
        self.mish1 = nn.Mish(inplace=False)
        self.norm1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(256, 128)
        self.mish2 = nn.Mish(inplace=False)
        self.norm2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(p=0.3)

        self.fc_out = nn.Linear(128, 2)  # Final output: 2 classes

    def forward(self, x):
        x = self.dropout1(self.norm1(self.mish1(self.fc1(x))))
        x = self.dropout2(self.norm2(self.mish2(self.fc2(x))))
        x = self.fc_out(x)
        return x

def get_model(device):
    """
    Loads the CVT backbone and attaches the custom classifier.

    Args:
        device (torch.device): Device to load the model on (CPU or CUDA)

    Returns:
        torch.nn.Module: Model ready for training or inference
    """
    model = CvtForImageClassification.from_pretrained('microsoft/cvt-13')

    # Replace classifier FIRST
    model.classifier = CustomClassifier()

    # THEN move entire model to device
    model.to(device)
    return model

