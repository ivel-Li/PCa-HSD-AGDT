"""
Base MRI Classifier (4-channel input backbone).
"""

import torch
import torch.nn as nn
import torchvision.models as models


class MRIClassifier(nn.Module):
    """
    4-channel input MRI classifier using ResNet backbone.
    """

    def __init__(self, num_classes=4, s=1, channels=4, pretrain_ckpt=None):
        super().__init__()
        self.s = s
        self.num_classes = num_classes

        # Load pretrained ResNet
        if pretrain_ckpt is not None and pretrain_ckpt != '':
            self.model = models.resnet50()
            state_dict = torch.load(pretrain_ckpt)
            self.model.load_state_dict(state_dict)
            print("ResNet50 loaded from offline weights.")
        else:
            self.model = models.resnet50(pretrained=True)

        # Replace first conv for 4-channel input
        self.model.conv1 = nn.Conv2d(
            channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Adjust fc layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes * s)

    def forward(self, x):
        out = self.model(x)
        if self.s != 1:
            out = out.view(-1, self.num_classes, self.s)
            out = torch.mean(out, dim=2)
        return out