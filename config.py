"""Default configuration for the current AGDT model."""

# ============== MODEL & Training Set =================
model_name = "AGDT"
trick_num = "postmask"
run = 42
DataParallel = False
device = "cuda:0"
num_epochs = 150
lr = 1e-5  # Learning rate
batch_size = 16  # 32/16 available for resnet and 2 for MedSAM
num_splits = 5  # 1 for no kfold
DATA_EXTAND = False
SupCon = False
seed = 42
num_workers = 4

# ProstateSAM3 transition-zone/peripheral-zone union mask.
LOAD_MASK = True
MASK_KEY = "post_train_mask"

# Paths (relative to project root)
HDF5_PATH = "./dataset/patients_dataset_postmask.h5"
