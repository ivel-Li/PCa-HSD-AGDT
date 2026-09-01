"""ViT-Large with inter-slice Transformer attention and no mask input."""

import os


model_name = "VITLargeAttentionClassifier"
trick_num = "nomask"
run = 42

DataParallel = False
device = "cuda:0"
num_epochs = 150
lr = 1e-5
batch_size = 16
num_splits = 5
DATA_EXTAND = False
SupCon = False
seed = 42
num_workers = 4

# This ablation and its data loader both explicitly exclude masks.
LOAD_MASK = False
MASK_KEY = "post_train_mask"
HDF5_PATH = "./dataset/patients_dataset_postmask.h5"

save_path = f"./run/{model_name}/{trick_num}/{run}"


def setup_directories():
    """Create the run directory and one output directory per fold."""
    os.makedirs(save_path, exist_ok=True)
    print("Directory created:", save_path)
    for fold in range(num_splits):
        fold_dir = os.path.join(save_path, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        print(f"Created folder: {fold_dir}")
