#!/usr/bin/env python3
"""
generate_post_train_mask.py

对于 patients_dataset_v1.0.h5 中的每个病人，读取 T2/ADC/DWI (16,224,224,1)，
对每张切片沿通道维 stack 成 3 通道图像，分别用 prompt "transition zone"
和 "peripheral zone" 运行 SAM3 推理，取并集为完整的前列腺 mask。

输出：新的 H5 文件 patients_dataset_postmask.h5（包含 v1.0 全部内容 + post_train_mask key）

用法：
    conda activate /data/shared/envs/sam3
    python dataset/generate_post_train_mask.py
"""

import os
import sys
import time
import shutil

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ── SAM3 imports ──────────────────────────────────────────────
sys.path.insert(0, "/data/users/lly/projects/sam3")
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ── Paths ─────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_H5 = os.path.join(PROJECT_ROOT, "dataset", "patients_dataset_v1.0.h5")
DST_H5 = os.path.join(PROJECT_ROOT, "dataset", "patients_dataset_postmask.h5")

SAM3_ROOT = "/data/users/lly/projects/sam3"
CHECKPOINT_PATH = "/data/users/lly/projects/sam3/experiments/prostate_formal_training_overlap/checkpoints/checkpoint_converted.pt"
BPE_PATH = os.path.join(SAM3_ROOT, "assets", "bpe_simple_vocab_16e6.txt.gz")

# ── SAM3 parameters ───────────────────────────────────────────
TEXT_PROMPTS = ["transition zone", "peripheral zone"]
CONFIDENCE_THRESHOLD = 0.3
RESOLUTION = 1008      # SAM3 input resolution
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── New H5 key ────────────────────────────────────────────────
MASK_KEY = "post_train_mask"


def normalize_volume_like_training(vol_slices: np.ndarray, clip_percentile: float = 99.5) -> np.ndarray:
    """
    对齐 ProstateSAM3/preprocess_overlap_prostate.py 的 load_and_normalize() 预处理方式：
      - volume-level 全局 99.5% percentile 截断 (而非 per-slice min-max)
      - 全局映射到 [0, 255]

    Args:
        vol_slices: (16, H, W) float32
        clip_percentile: 截断百分位数，与训练一致 (default 99.5)
    Returns:
        (16, H, W) uint8 [0, 255]
    """
    vmin = 0.0
    vmax = float(np.percentile(vol_slices, clip_percentile))
    vol_clipped = np.clip(vol_slices, vmin, vmax)
    if vmax > vmin:
        vol_norm = (vol_clipped - vmin) / (vmax - vmin) * 255.0
    else:
        vol_norm = np.zeros_like(vol_clipped)
    return vol_norm.astype(np.uint8)


def build_sam3_processor() -> Sam3Processor:
    """Initialize SAM3 model (overlap checkpoint) and return the processor."""
    print(f"Using device: {DEVICE}")

    # Enable TF32 on Ampere GPUs for performance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if DEVICE == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    # BPE path fallback
    bpe_path = BPE_PATH
    if not os.path.exists(bpe_path):
        import pkg_resources
        bpe_path = pkg_resources.resource_filename(
            "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"
        )

    print(f"Loading SAM3 model from checkpoint: {CHECKPOINT_PATH}")
    print(f"BPE path: {bpe_path}")

    model = build_sam3_image_model(
        bpe_path=bpe_path,
        device=DEVICE,
        checkpoint_path=CHECKPOINT_PATH,
        load_from_HF=False,
        eval_mode=True,
        enable_segmentation=True,
    )

    processor = Sam3Processor(
        model,
        resolution=RESOLUTION,
        device=DEVICE,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )

    print("SAM3 model loaded successfully.")
    return processor


def process_patient_multimodal(
    processor: Sam3Processor,
    t2_data: np.ndarray,   # (16, 224, 224, 1) float32
    adc_data: np.ndarray,  # (16, 224, 224, 1) float32
    dwi_data: np.ndarray,  # (16, 224, 224, 1) float32
) -> np.ndarray:
    """
    Run SAM3 on every slice (T2+ADC+DWI stacked as 3-channel) with two
    text prompts and union their masks.

    预处理流水线已对齐 ProstateSAM3/preprocess_overlap_prostate.py：
      1. volume-level 99.5% percentile 截断 + 全局映射 [0,255]
      2. 顺时针旋转 90° 对齐训练数据的图像朝向
      3. 推理后的 mask 逆时针旋转 90° 恢复原始方向

    Returns:
        mask_volume: (16, 224, 224, 1) uint8, values {0, 1}.
    """
    num_slices = t2_data.shape[0]
    h, w = t2_data.shape[1], t2_data.shape[2]
    mask_volume = np.zeros((num_slices, h, w, 1), dtype=np.uint8)

    # ── Step 1: Volume-level 99.5% percentile normalization ──
    # Align with ProstateSAM3/preprocess_overlap_prostate.py:load_and_normalize()
    t2_vol = normalize_volume_like_training(t2_data[:, :, :, 0])  # (16, H, W) uint8
    adc_vol = normalize_volume_like_training(adc_data[:, :, :, 0])
    dwi_vol = normalize_volume_like_training(dwi_data[:, :, :, 0])

    for z in range(num_slices):
        # Extract 2D slices (already normalized globally)
        t2_uint8 = t2_vol[z]    # (H, W) uint8
        adc_uint8 = adc_vol[z]  # (H, W) uint8
        dwi_uint8 = dwi_vol[z]  # (H, W) uint8

        # ── Step 2: Rotate 90° clockwise to align with training data ──
        # ProstateSAM3/dataset 下的 2D PNG 朝向为原始 NIfTI 顺时针旋转 90°
        t2_rot = np.rot90(t2_uint8, k=-1)    # clockwise
        adc_rot = np.rot90(adc_uint8, k=-1)
        dwi_rot = np.rot90(dwi_uint8, k=-1)

        # Stack as 3-channel RGB image: T2=R, ADC=G, DWI=B (same as training)
        rgb_img = np.stack([t2_rot, adc_rot, dwi_rot], axis=-1)  # (W, H, 3)
        pil_img = Image.fromarray(rgb_img)

        # ── Run SAM3 for each prompt and accumulate union mask ──
        union_mask_rot = np.zeros((rgb_img.shape[0], rgb_img.shape[1]), dtype=np.uint8)

        for prompt in TEXT_PROMPTS:
            inference_state = processor.set_image(pil_img)
            inference_state = processor.set_text_prompt(
                state=inference_state, prompt=prompt
            )

            masks = inference_state.get("masks")      # (N, 1, H', W') bool tensor
            scores = inference_state.get("scores")    # (N,) float tensor

            if masks is not None and len(masks) > 0 and scores is not None and len(scores) > 0:
                best_idx = int(scores.argmax().item())
                mask_np = masks[best_idx, 0].cpu().numpy().astype(np.uint8)  # (H', W')
                union_mask_rot = np.maximum(union_mask_rot, mask_np)  # union (logical OR)

        # ── Step 3: Rotate mask back counter-clockwise ──
        union_mask = np.rot90(union_mask_rot, k=1)  # counter-clockwise → (H, W)

        mask_volume[z, :, :, 0] = union_mask

    return mask_volume


def main():
    # ── Validate inputs ───────────────────────────────────────
    if not os.path.exists(SRC_H5):
        print(f"Error: Source H5 file not found: {SRC_H5}")
        sys.exit(1)
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: SAM3 checkpoint not found: {CHECKPOINT_PATH}")
        sys.exit(1)

    # ── Copy the original H5 to the new file ──────────────────
    if not os.path.exists(DST_H5):
        print(f"Copying {SRC_H5} → {DST_H5} ...")
        shutil.copy2(SRC_H5, DST_H5)
        print("Copy done.")
    else:
        print(f"Destination {DST_H5} already exists, will add key to it.")

    # ── Build SAM3 ────────────────────────────────────────────
    processor = build_sam3_processor()

    # ── Open H5 and process ───────────────────────────────────
    print(f"\nOpening H5 file: {DST_H5}")
    f = h5py.File(DST_H5, "r+")

    patient_ids = sorted(f.keys(), key=int)
    print(f"Total patients in H5: {len(patient_ids)}")

    # Check which patients already have the mask key
    already_done = 0
    for pid in patient_ids:
        if MASK_KEY in f[pid]:
            already_done += 1
    print(f"Patients with existing '{MASK_KEY}': {already_done}")
    if already_done > 0:
        print("(These will be skipped unless you remove them manually.)\n")

    start_time = time.time()
    success_count = 0
    skip_count = 0
    fail_count = 0

    for pid in tqdm(patient_ids, desc="Processing patients"):
        grp = f[pid]

        # Skip if mask already exists
        if MASK_KEY in grp:
            skip_count += 1
            continue

        # Read T2 / ADC / DWI
        if "T2" not in grp or "ADC" not in grp or "DWI" not in grp:
            tqdm.write(f"[WARN] Patient {pid}: missing modality, skipping.")
            fail_count += 1
            continue

        t2_data = grp["T2"][:]    # (16, 224, 224, 1) float32
        adc_data = grp["ADC"][:]  # (16, 224, 224, 1) float32
        dwi_data = grp["DWI"][:]  # (16, 224, 224, 1) float32

        # Generate SAM3 masks
        try:
            masks = process_patient_multimodal(processor, t2_data, adc_data, dwi_data)
        except Exception as e:
            tqdm.write(f"[ERROR] Patient {pid}: SAM3 inference failed: {e}")
            import traceback
            tqdm.write(traceback.format_exc())
            fail_count += 1
            continue

        # Write mask dataset into the H5 group
        grp.create_dataset(
            MASK_KEY,
            data=masks,
            compression="gzip",
            compression_opts=4,
        )
        success_count += 1

    f.close()

    elapsed = time.time() - start_time
    total = success_count + skip_count + fail_count
    print(f"\n{'='*50}")
    print(f"Done! Processed {total} patients:")
    print(f"  Success: {success_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Failed:  {fail_count}")
    print(f"  Time:    {elapsed:.1f}s total, "
          f"{elapsed / max(total, 1):.2f}s per patient")
    print(f"\nNew H5 key: /<patient_index>/{MASK_KEY}")
    print(f"  shape: (16, 224, 224, 1), dtype: uint8, values: {{0, 1}}")
    print(f"  Note: mask = union of 'transition zone' and 'peripheral zone' prompts")
    print(f"\nOutput file: {DST_H5}")


if __name__ == "__main__":
    main()
