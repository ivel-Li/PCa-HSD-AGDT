"""AGDT training configuration using MedicalSAM3 prostate masks."""

import os


model_name = "AGDT"
trick_num = "medicalsam3mask"
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

LOAD_MASK = True
MASK_KEY = "medicalsam3"
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
