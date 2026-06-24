"""
BioVIT / OmniRad Classifier (specialized ViT variants from BiomedCLIP or OmniRad).
Supports auto-download of pretrained weights when local checkpoint not found.
"""

import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from config import device


class BioVITClassifier(nn.Module):
    def __init__(
        self,
        vit_name='Biomedclip_vit',
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

        # Build backbone and load weights
        self.vit, embed_dim = self._build_backbone(vit_name, vit_ckpt)

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

    def _build_backbone(self, vit_name, vit_ckpt):
        """Build backbone and load weights. Returns (model, embed_dim)."""

        if vit_name == "Biomedclip_vit":
            embed_dim = 512
            # Priority 1: local checkpoint
            if vit_ckpt is not None and os.path.isfile(vit_ckpt):
                model = torch.load(vit_ckpt, map_location=device, weights_only=False)
                print(f"[BioVITClassifier] Loaded BiomedCLIP weights from {vit_ckpt}")
                return model, embed_dim
            # Fallback: random init + warning
            warnings.warn(
                f"[BioVITClassifier] BiomedCLIP checkpoint not found at {vit_ckpt}.\n"
                f"  Falling back to random initialization. "
                f"Download from: https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
            # Build a default ViT-B/16 for BiomedCLIP shape compatibility
            model = timm.create_model(
                "vit_base_patch16_224", pretrained=False, num_classes=0, global_pool='avg'
            )
            return model, embed_dim

        elif vit_name == "Omnirad_vit":
            # Try building backbone first, then load weights
            model = timm.create_model(
                "hf_hub:Snarcy/OmniRad-base", pretrained=False
            )
            embed_dim = model.num_features

            # Priority 1: local checkpoint
            if vit_ckpt is not None and os.path.isfile(vit_ckpt):
                state = torch.load(vit_ckpt, map_location="cpu", weights_only=True)
                model.load_state_dict(state, strict=True)
                print(f"[BioVITClassifier] Loaded OmniRad weights from {vit_ckpt}")
            else:
                # Priority 2: auto-download via HuggingFace hub
                try:
                    print("[BioVITClassifier] Auto-downloading OmniRad pretrained weights ...")
                    model = timm.create_model(
                        "hf_hub:Snarcy/OmniRad-base", pretrained=True
                    )
                    print("[BioVITClassifier] Auto-downloaded OmniRad pretrained weights")
                except Exception as e:
                    warnings.warn(
                        f"[BioVITClassifier] Could not auto-download OmniRad weights: {e}\n"
                        f"  Falling back to random initialization."
                    )

            model.reset_classifier(0)
            return model, embed_dim

        else:
            raise ValueError(f"Unknown vit_name: {vit_name}")

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