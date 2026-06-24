"""Configuration for SWIN-Split project."""
import os

# ============== MODEL & Training Set =================
model_name = "LSDT-Large"
run = 1  
trick_num = 5
DataParallel = False
device = "cuda:0"
num_epochs = 150
batch_size = 16  # 32/16 available for resnet and 2 for MedSAM
num_splits = 5  # 1 for no kfold
DATA_EXTAND = False
SupCon = False
seed = 42
num_workers = 4

# Paths (relative to project root)
HDF5_PATH = "./dataset/patients_dataset_v1.0.h5"

# Setup save path
# 1.2 for k-fold, 1.3 for data_extand, 1.4 for contrastive learning, 1.5 for v1.0dataset
save_path = f"./run/{model_name}/{trick_num}/{run}"


def setup_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(save_path, exist_ok=True)
    print("Directory created:", save_path)
    for i in range(num_splits):
        fold_dir = os.path.join(save_path, f"fold_{i}")
        os.makedirs(fold_dir, exist_ok=True)
        print(f"Created folder: {fold_dir}")
