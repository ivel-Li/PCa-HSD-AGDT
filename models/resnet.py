"""
ResNet Classifier with optional fusion (transformer-based slice fusion).
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F


class ResNetClassifier(nn.Module):
    def __init__(
        self,
        num_classes=4,
        s=1,
        channels=3,
        pretrain_ckpt=None,
        use_mask=False,
        fusion=None,
        proj_dim=128,
        nhead=8,
        num_layers=2,
        num_slices=16
    ):
        super().__init__()

        self.use_mask = use_mask
        self.fusion = fusion
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

        # Replace first conv for 3/4-channel input
        self.model.conv1 = nn.Conv2d(
            channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Remove classification head
        self.model.fc = nn.Identity()
        embed_dim = 2048

        # Position encoding
        if fusion is not None:
            self.pos_embed = nn.Parameter(
                torch.randn(1, num_slices, embed_dim)
            )

        # Transformer fusion
        if fusion == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )

        # CNN fusion
        if fusion == "cnn":
            self.slice_cnn = nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.ReLU(),
            )

        # Concatenation fusion
        if fusion == "cat":
            self.catclassifier = nn.Sequential(
                nn.Linear(num_slices * embed_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes),
            )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_classes),
        )

        # Slice-level classifier
        self.fc = nn.Linear(embed_dim, num_classes)

        # Projection head
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, adc, dwi, t2, mask=None):
        """
        adc, dwi, t2: (B, S, 1, H, W)
        mask:         (B, S, 1, H, W) or None
        """
        B, S, _, H, W = adc.shape

        if self.use_mask and mask is not None:
            mask_proc = mask.float()
            adc = adc * mask_proc
            dwi = dwi * mask_proc
            t2 = t2 * mask_proc

        x = torch.cat([adc, dwi, t2], dim=2)  # (B, S, 3, H, W)
        x = x.view(B * S, 3, H, W)

        # ResNet features
        slice_features = self.model(x)  # (B*S, 2048)

        if self.fusion is None:
            slice_logits = self.fc(slice_features)
            slice_logits = slice_logits.view(B, S, -1)
            output = torch.sum(slice_logits, dim=1)

        elif self.fusion == "transformer":
            feat = slice_features.view(B, S, -1)
            feat = feat + self.pos_embed[:, :S, :]
            feat = self.transformer(feat)
            pooled = feat.mean(dim=1)
            output = self.classifier(pooled)

        elif self.fusion == "pool":
            feat = slice_features.view(B, S, -1)
            pooled = feat.mean(dim=1)
            output = self.classifier(pooled)

        elif self.fusion == "cnn":
            feat = slice_features.view(B, S, -1)
            feat = feat.permute(0, 2, 1)
            feat = self.slice_cnn(feat)
            pooled = feat.mean(dim=2)
            output = self.classifier(pooled)

        elif self.fusion == "cat":
            feat = slice_features.view(B, S, -1)
            feat = feat.flatten(start_dim=1)
            output = self.catclassifier(feat)

        else:
            raise ValueError("Invalid fusion mode")

        proj = F.normalize(self.proj_head(slice_features), dim=1)
        return output, proj