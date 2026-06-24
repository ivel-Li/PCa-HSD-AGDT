#!/usr/bin/env python3
"""
MaskProcess.py — Generate SAM3 prostate pseudo-labels from T2_res stored in the H5 dataset.

For each patient group in patients_dataset_v1.0.h5:
  - Reads T2_res:      (16, 224, 224, 1) float32, normalized in [0,1]
  - For each 2D slice, runs SAM3 with text prompt "prostate" and confidence
    threshold 0.6 to produce a binary mask.
  - Stacks the 16 slice masks back to (16, 224, 224, 1) uint8.
  - Inserts /<patient_index>/T2_res_sam3_text_mask_0.6 into the same H5 file.

Usage:
    conda activate /data/shared/envs/sam3
    python projects/PCa-HSD-LSDT/dataset/MaskProcess.py

Requirements:
    - SAM3 installed at /data/users/lly/projects/sam3
    - SAM3 checkpoint at ~/sam3.pt
    - patients_dataset_v1.0.h5 under the dataset/ directory
"""

import os
import sys
import time

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
H5_PATH = os.path.join(PROJECT_ROOT, "dataset", "patients_dataset_v1.0.h5")

SAM3_ROOT = "/data/users/lly/projects/sam3"
CHECKPOINT_PATH = os.path.expanduser("~/sam3.pt")
BPE_PATH = os.path.join(SAM3_ROOT, "assets", "bpe_simple_vocab_16e6.txt.gz")

# ── SAM3 parameters ───────────────────────────────────────────
TEXT_PROMPT = "prostate"
CONFIDENCE_THRESHOLD = 0.6
RESOLUTION = 1008      # SAM3 input resolution
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── New H5 key ────────────────────────────────────────────────
MASK_KEY = "T2_res_sam3_text_mask_0.6"


def normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    """Min-max normalize a 2D slice from any dtype to uint8 [0, 255]."""
    lo = float(slice_2d.min())
    hi = float(slice_2d.max())
    if hi - lo < 1e-6:
        return np.zeros_like(slice_2d, dtype=np.uint8)
    normalized = (slice_2d.astype(np.float32) - lo) / (hi - lo) * 255.0
    return np.clip(normalized, 0, 255).astype(np.uint8)


def build_sam3_processor() -> Sam3Processor:
    """Initialize SAM3 model and return the processor."""
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
    )

    processor = Sam3Processor(
        model,
        resolution=RESOLUTION,
        device=DEVICE,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )

    print("SAM3 model loaded successfully.")
    return processor


def process_patient_t2_res(
    processor: Sam3Processor,
    t2_res_data: np.ndarray,
) -> np.ndarray:
    """
    Run SAM3 on every slice in T2_res and produce a binary mask volume.

    Args:
        processor: Initialized Sam3Processor.
        t2_res_data: (16, 224, 224, 1) float32 in [0, 1].

    Returns:
        mask_volume: (16, 224, 224, 1) uint8, values {0, 1}.
    """
    num_slices = t2_res_data.shape[0]
    h, w = t2_res_data.shape[1], t2_res_data.shape[2]
    mask_volume = np.zeros((num_slices, h, w, 1), dtype=np.uint8)

    for z in range(num_slices):
        # Extract 2D slice and squeeze channel dim -> (H, W)
        slice_2d = t2_res_data[z, :, :, 0]  # float32 in [0, 1]

        # Normalize to uint8 [0, 255]
        slice_uint8 = normalize_slice(slice_2d)

        # Convert grayscale to RGB (SAM3 expects 3-channel PIL image)
        pil_img = Image.fromarray(slice_uint8, mode="L").convert("RGB")

        # ── SAM3 inference ──
        inference_state = processor.set_image(pil_img)
        inference_state = processor.set_text_prompt(
            state=inference_state, prompt=TEXT_PROMPT
        )

        # Extract best mask
        masks = inference_state.get("masks")      # (N, 1, H, W) bool tensor
        scores = inference_state.get("scores")    # (N,) float tensor

        if masks is not None and len(masks) > 0 and scores is not None and len(scores) > 0:
            best_idx = int(scores.argmax().item())
            mask_np = masks[best_idx, 0].cpu().numpy().astype(np.uint8)  # (H, W)
            mask_volume[z, :, :, 0] = mask_np
        else:
            # No confident detection → all zeros for this slice
            mask_volume[z, :, :, 0] = np.zeros((h, w), dtype=np.uint8)

    return mask_volume


def main():
    # ── Validate inputs ───────────────────────────────────────
    if not os.path.exists(H5_PATH):
        print(f"Error: H5 file not found: {H5_PATH}")
        sys.exit(1)
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: SAM3 checkpoint not found: {CHECKPOINT_PATH}")
        sys.exit(1)

    # ── Build SAM3 ────────────────────────────────────────────
    processor = build_sam3_processor()

    # ── Open H5 and process ───────────────────────────────────
    print(f"\nOpening H5 file: {H5_PATH}")
    f = h5py.File(H5_PATH, "r+")

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

        # Read T2_res data
        if "T2" not in grp:
            tqdm.write(f"[WARN] Patient {pid}: no T2_res key, skipping.")
            fail_count += 1
            continue

        t2_res = grp["T2"][:]  # (16, 224, 224, 1) float32

        # Generate SAM3 masks
        try:
            masks = process_patient_t2_res(processor, t2_res)
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


if __name__ == "__main__":
    main()