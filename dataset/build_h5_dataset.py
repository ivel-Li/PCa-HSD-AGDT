"""
build_h5_dataset.py

将 dataset/ 下按类别/设备/病人的 MRI (ADC/DWI/T2 .nii/.nii.gz) 预处理并保存为单个 HDF5 文件。

标签映射（在此脚本中固定为）：
0: normal_prostate
1: benign_prostate_cancer
2: non_significant_prostate_cancer
3: significant_prostate_cancer

输出 HDF5 结构（每个病人一个 group，name 为 patient_index）：
    /<patient_index>/
        label   : int64 scalar (0..3)
        ADC     : (16,224,224,1) float32 (values normalized in [0,1])
        DWI     : (16,224,224,1) float32
        T2      : (16,224,224,1) float32
        attrs: orig_shape_ADC, orig_shape_DWI, orig_shape_T2 (strings)

用法：
    python dataset/build_h5_dataset.py
"""

import os
import numpy as np
import SimpleITK as sitk
from skimage import transform
import h5py
from tqdm import tqdm
import glob
import psutil
import gc


# ---------- 配置区 ----------
DATA_ROOT = "/data/users/lly/projects/PCa-HSD-LSDT/dataset/v1.0"  # 原始 NIfTI 路径
OUTPUT_H5 = "patients_dataset_v1.0.h5"  # 输出 h5 文件
MODALITIES = ["DWI", "ADC", "T2"]       # 所需模态（脚本在 patient dir 下查找 <MODALITY>.nii/.nii.gz）
MID_SLICES = 16                         # 每个模态取的中间切片数
TARGET_SIZE = (224, 224)                # 重采样目标尺寸 HxW
BATCH_SIZE = 20                         # 批大小，控制内存
# --------------------------------

# label -> 类别名 的映射
LABEL_MAP = {
    "normal_prostate": 0,
    "benign_prostate_cancer": 1,
    "non_significant_prostate_cancer": 2,
    "significant_prostate_cancer": 3,
}

NIIFMT = [".nii", ".nii.gz"]


def find_modality_file(patient_dir, modality):
    """在 patient_dir 下查找 modality 的 .nii/.nii.gz 文件（大小写不敏感）。"""
    for ext in NIIFMT:
        p = os.path.join(patient_dir, modality + ext)
        if os.path.exists(p):
            return p
    files = glob.glob(os.path.join(patient_dir, "*"))
    for f in files:
        fname = os.path.basename(f).lower()
        if fname.startswith(modality.lower()) and (fname.endswith(".nii") or fname.endswith(".nii.gz")):
            return f
    return None


def read_vol(path):
    """读取 NIfTI 并返回 (numpy_volume, spacing) 其中 volume shape=(Z,H,W)。"""
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    return arr, img.GetSpacing()


def take_center_slices(vol, num_slices):
    """
    从 vol (Z,H,W) 取中间 num_slices 张切片。
    若 Z < num_slices，以重复最后一张补齐。
    """
    Z = vol.shape[0]
    if Z >= num_slices:
        start = (Z - num_slices) // 2
        return vol[start:start + num_slices]
    else:
        pad_needed = num_slices - Z
        pads = np.repeat(vol[-1][None, ...], pad_needed, axis=0)
        return np.concatenate([vol, pads], axis=0)


def preprocess_slice_gray_to_1ch(slice2d, target_size):
    """
    单张灰度切片处理流水线：
      - 复制为 1 通道 → (H,W,1)
      - cubic 插值重采样到 target_size
      - per-slice min-max 归一化到 [0,1]
    返回 float32 数组 shape=(H,W,1)。
    """
    sl = slice2d.astype(np.float32)
    sl1 = np.repeat(sl[:, :, None], 1, axis=-1)  # (H, W, 1)
    out_shape = (target_size[0], target_size[1], 1)
    resized = transform.resize(
        sl1,
        out_shape,
        order=3,
        preserve_range=True,
        anti_aliasing=True,
        mode="constant",
    )
    mn, mx = resized.min(), resized.max()
    if mx - mn > 1e-8:
        norm = (resized - mn) / (mx - mn)
    else:
        norm = np.zeros_like(resized, dtype=np.float32)
    return norm.astype(np.float32)


def process_patient(patient_dir):
    """
    处理单个病人目录：
      - 读取 ADC/DWI/T2；缺失则返回 None
      - 提取中间切片并预处理每张切片
      - 返回 dict { 'vols': {mod: (16,224,224,1)}, 'orig_shapes', 'spacings', 'machine' }
    """
    vols = {}
    orig_shapes = {}
    spacings = {}
    machine = os.path.basename(os.path.dirname(patient_dir))
    for mod in MODALITIES:
        fpath = find_modality_file(patient_dir, mod)
        if fpath is None:
            print(f"[WARN] {patient_dir} missing {mod}, skip.")
            return None
        vol, spacing = read_vol(fpath)
        orig_shapes[mod] = vol.shape
        spacings[mod] = spacing
        vol16 = take_center_slices(vol, MID_SLICES)
        processed = []
        for i in range(vol16.shape[0]):
            processed.append(preprocess_slice_gray_to_1ch(vol16[i], TARGET_SIZE))
        vols[mod] = np.stack(processed, axis=0)
    return {"vols": vols, "orig_shapes": orig_shapes, "spacings": spacings, "machine": machine}


def gather_patients(root_dir):
    """遍历 root_dir 收集所有包含 .nii 文件的目录作为病人目录。"""
    entries = []
    for label_name in LABEL_MAP.keys():
        class_dir = os.path.join(root_dir, label_name)
        if not os.path.isdir(class_dir):
            print(f"[WARN] class dir not found: {class_dir}")
            continue
        for root, dirs, files in os.walk(class_dir):
            has_nii = any(f.lower().endswith(".nii") or f.lower().endswith(".nii.gz") for f in files)
            if has_nii:
                entries.append((root, label_name))
    return entries


def memory_usage():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def main():
    patients = gather_patients(DATA_ROOT)
    print(f"Found {len(patients)} candidate patient folders, memory: {memory_usage():.1f}MB")

    total_batches = (len(patients) + BATCH_SIZE - 1) // BATCH_SIZE
    global_patient_index = 0

    with h5py.File(OUTPUT_H5, "w") as f:
        for batch_idx in range(total_batches):
            batch_start = batch_idx * BATCH_SIZE
            batch_end = min((batch_idx + 1) * BATCH_SIZE, len(patients))
            batch_patients = patients[batch_start:batch_end]

            print(f"\nBatch {batch_idx + 1}/{total_batches} (patients {batch_start}-{batch_end-1})")
            batch_processed = []
            for pdir, lbl in tqdm(batch_patients, desc=f"Batch {batch_idx+1}"):
                try:
                    res = process_patient(pdir)
                    if res is not None:
                        batch_processed.append((pdir, lbl, res))
                except Exception as e:
                    print(f"[ERROR] {pdir}: {e}")

            for pdir, lbl, res in batch_processed:
                grp = f.create_group(str(global_patient_index))
                grp.create_dataset("label", data=np.int64(LABEL_MAP[lbl]))
                grp.attrs["machine_name"] = res["machine"]
                for mod in MODALITIES:
                    data = res["vols"][mod]
                    grp.create_dataset(mod, data=data, compression="gzip", compression_opts=4)
                    grp.attrs[f"orig_shape_{mod}"] = str(res["orig_shapes"][mod])
                    grp.attrs[f"spacing_{mod}"] = str(res["spacings"][mod])
                grp.attrs["patient_dir"] = pdir
                global_patient_index += 1

            del batch_processed
            gc.collect()
            print(f"  → {global_patient_index} patients saved, memory: {memory_usage():.1f}MB")

    print(f"\nDone! {global_patient_index} patients → {OUTPUT_H5}")


if __name__ == "__main__":
    main()