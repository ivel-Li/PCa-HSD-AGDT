"""
Model factory – mirrors build_model() from the SWIN-Split.ipynb notebook.
Maps model name strings to model class + constructor kwargs.

Naming conventions:
  - AGDT: mask-guided ViT-Large with inter-slice Transformer aggregation.
  - Original names (e.g. VITClassifier, ResNetClassifier, Omnirad_vit):
    use_mask=False, fusion=None (plain classifier).
  - LSDT-* names (e.g. LSDT-Base, LSDT-ResNet50, LSDT-Omnirad):
    use_mask=True, fusion="transformer" (full LSDT pipeline).
"""

from .vit import VITClassifier
from .bio_vit import BioVITClassifier
from .resnet import ResNetClassifier
from .base import MRIClassifier


def build_model(name):
    """Build a model by name."""

    # LSDT-Large is retained so historical configs and checkpoints still load.
    if name in {"AGDT", "LSDT-Large"}:
        return VITClassifier(
            vit_name="vit_large_patch16_224",
            vit_ckpt=None,
            use_mask=True,
            fusion="transformer",
        )

    # ═════════════════════════════════════════════════════════════════
    # LSDT models: use_mask=True, fusion="transformer"
    # ═════════════════════════════════════════════════════════════════
    if name == "LSDT-ResNet50":
        return ResNetClassifier(use_mask=True, fusion="transformer")
    if name == "LSDT-Swin-T":
        return VITClassifier(
            vit_name="swin_base_patch4_window7_224",
            vit_ckpt=None,
            use_mask=True,
            fusion="transformer",
        )
    if name == "LSDT-Base":
        return VITClassifier(vit_ckpt=None, use_mask=True, fusion="transformer")
    if name == "LSDT-BiomedCLIP":
        return BioVITClassifier(
            vit_name="Biomedclip_vit",
            vit_ckpt=None,
            use_mask=True,
            fusion="transformer",
        )
    if name == "LSDT-Omnirad":
        return BioVITClassifier(
            vit_name="Omnirad_vit",
            vit_ckpt=None,
            use_mask=True,
            fusion="transformer",
        )

    # ═════════════════════════════════════════════════════════════════
    # Default models (original names): use_mask=False, fusion=None
    # ═════════════════════════════════════════════════════════════════
    if name == "ResNetClassifier":
        return ResNetClassifier()
    if name == "MRIClassifier":
        return MRIClassifier()
    if name == "VITClassifier":
        return VITClassifier(vit_ckpt=None)
    if name == "VITLargeClassifier":
        return VITClassifier(vit_name="vit_large_patch16_224", vit_ckpt=None)
    if name == "VITLargeAttentionClassifier":
        return VITClassifier(
            vit_name="vit_large_patch16_224",
            vit_ckpt=None,
            use_mask=False,
            fusion="transformer",
        )
    if name == "VITLarge21kClassifier":
        return VITClassifier(
            vit_name="vit_large_patch16_224_in21k",
            vit_ckpt=None,
        )
    if name == "VITHugeClassifier":
        return VITClassifier(vit_name="vit_huge_patch14_224", vit_ckpt=None)
    if name == "SWINClassifier":
        return VITClassifier(
            vit_name="swin_base_patch4_window7_224",
            vit_ckpt=None,
        )
    if name == "Biomedclip_vit":
        return BioVITClassifier(vit_name="Biomedclip_vit", vit_ckpt=None)
    if name == "Omnirad_vit":
        return BioVITClassifier(vit_name="Omnirad_vit", vit_ckpt=None)

    # ═════════════════════════════════════════════════════════════════
    # Legacy aliases (backward compatible, will be deprecated)
    # ═════════════════════════════════════════════════════════════════
    if name == "ResNetfusion":
        return ResNetClassifier(use_mask=True, fusion="transformer")
    if name == "VITBasefusion":
        return VITClassifier(vit_ckpt=None, use_mask=True, fusion="transformer")
    if name == "VITLargefusion":
        return VITClassifier(
            vit_name="vit_large_patch16_224",
            vit_ckpt=None,
            use_mask=True,
            fusion="cat",
        )
    if name == "LSDTNofusion":
        return VITClassifier(
            vit_name="vit_large_patch16_224",
            vit_ckpt=None,
            use_mask=True,
            fusion=None,
        )
    if name == "LSDTcatfusion":
        return VITClassifier(
            vit_name="vit_large_patch16_224",
            vit_ckpt=None,
            use_mask=True,
            fusion="cat",
        )
    if name == "LSDTmeanfusion":
        return VITClassifier(
            vit_name="vit_large_patch16_224",
            vit_ckpt=None,
            use_mask=True,
            fusion="pool",
        )
    if name == "LSDTcnnfusion":
        return VITClassifier(
            vit_name="vit_large_patch16_224",
            vit_ckpt=None,
            use_mask=True,
            fusion="cnn",
        )
    if name == "VITHugefusion":
        return VITClassifier(
            vit_name="vit_huge_patch14_224",
            vit_ckpt=None,
            use_mask=True,
            fusion="cnn",
        )
    if name == "VITHugeMaefusion":
        return VITClassifier(
            vit_name="vit_huge_patch14_224.mae",
            vit_ckpt=None,
            use_mask=True,
            fusion="transformer",
        )
    if name == "SWINfusion":
        return VITClassifier(
            vit_name="swin_base_patch4_window7_224",
            vit_ckpt=None,
            use_mask=True,
            fusion="transformer",
        )
    if name == "Biomedclip_vit_fusion":
        return BioVITClassifier(
            vit_name="Biomedclip_vit",
            vit_ckpt=None,
            use_mask=True,
            fusion="transformer",
        )
    if name == "Omnirad_vit_fusion":
        return BioVITClassifier(
            vit_name="Omnirad_vit",
            vit_ckpt=None,
            use_mask=True,
            fusion="transformer",
        )

    raise ValueError(f"Unknown model name: {name}")
