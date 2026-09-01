"""No-mask ViT-Large ablation with inter-slice Transformer aggregation."""

from config import *  # noqa: F401,F403

model_name = "VITLargeAttentionClassifier"
trick_num = "nomask"
LOAD_MASK = False
