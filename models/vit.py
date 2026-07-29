"""
VIT Classifier for 3-channel (adc, dwi, t2) + optional mask.
Supports auto-download of pretrained weights when local checkpoint not found.
"""

import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class VITClassifier(nn.Module):
    def __init__(
        self,
        vit_name='vit_base_patch16_224',
        num_classes=4,
        pretrained=False,
        vit_ckpt=None,
        use_mask=False,
        fusion=None,
        proj_dim=128,
        nhead=8,
        num_layers=2,
        num_slices=16
    ):
        super().__init__()

        self.use_mask = use_mask

        # ViT backbone (remove classification head)
        self.vit = timm.create_model(
            vit_name,
            pretrained=False,
            num_classes=0,
            global_pool='avg',
        )

        embed_dim = self.vit.num_features
        self.fusion = fusion

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

        # Projection head for contrastive loss
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
        )

        # Load pretrained weights
        self._load_weights(vit_name, vit_ckpt)

    def _load_weights(self, vit_name, vit_ckpt):
        """Load pretrained weights: local file > timm auto-download > random init."""
        # Priority 1: local checkpoint file
        if vit_ckpt is not None and os.path.isfile(vit_ckpt):
            state_dict = torch.load(vit_ckpt, map_location='cpu')
            state_dict.pop('head.weight', None)
            state_dict.pop('head.bias', None)
            self.vit.load_state_dict(state_dict, strict=False)
            print(f"[VITClassifier] Loaded pretrained weights from {vit_ckpt}")
            return

        # Priority 2: timm auto-download (pretrained=True)
        try:
            print(f"[VITClassifier] Auto-downloading pretrained weights for {vit_name} ...")
            pretrained_model = timm.create_model(
                vit_name, pretrained=True, num_classes=0, global_pool='avg'
            )
            self.vit.load_state_dict(pretrained_model.state_dict(), strict=False)
            print(f"[VITClassifier] Auto-downloaded pretrained weights for {vit_name}")
            return
        except Exception as e:
            warnings.warn(
                f"[VITClassifier] Could not auto-download pretrained weights for {vit_name}: {e}\n"
                f"  Falling back to random initialization."
            )

    def forward(self, adc, dwi, t2, mask=None):
        """
        adc, dwi, t2: (B, S, 1, H, W)
        mask:         (B, S, 1, H, W) or None
        """
        B, S, _, H, W = adc.shape

        # Mask processing
        if self.use_mask and mask is not None:
            mask_proc = mask.float()
            adc = adc * mask_proc
            dwi = dwi * mask_proc
            t2 = t2 * mask_proc

        # Concatenate modalities (channel dim)
        x = torch.cat([adc, dwi, t2], dim=2)  # (B, S, 3, H, W)

        # Slice as batch
        x = x.view(B * S, 3, H, W)  # (B*S, 3, H, W)

        # ViT slice features
        slice_features = self.vit(x)  # (B*S, embed_dim)

        if self.fusion is None:
            # Slice logits
            slice_logits = self.fc(slice_features)  # (B*S, num_classes)
            slice_logits = slice_logits.view(B, S, -1)  # (B, S, num_classes)
            output = torch.sum(slice_logits, dim=1)  # (B, num_classes)

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
            feat = feat.permute(0, 2, 1)  # (B, E, S)
            feat = self.slice_cnn(feat)   # (B, E, S)
            pooled = feat.mean(dim=2)     # (B, E)
            output = self.classifier(pooled)

        elif self.fusion == "cat":
            feat = slice_features.view(B, S, -1)
            feat = feat.flatten(start_dim=1)
            output = self.catclassifier(feat)

        else:
            raise ValueError("Invalid fusion mode")

        proj = F.normalize(self.proj_head(slice_features), dim=1)
        return output, proj