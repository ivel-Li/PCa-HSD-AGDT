#!/usr/bin/env python3
"""Export H5 T2 volumes as per-slice normalized uint8 NumPy arrays."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


DEFAULT_H5 = Path(__file__).with_name("patients_dataset_postmask.h5")
DEFAULT_OUTPUT = Path("/tmp/pcahsd_t2_sam3_input")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-key", default="T2")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def natural_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def normalize_slices(volume: np.ndarray) -> np.ndarray:
    if volume.ndim == 4 and volume.shape[-1] == 1:
        volume = volume[..., 0]
    if volume.ndim != 3:
        raise ValueError(f"Expected (slices, height, width[, 1]), got {volume.shape}")

    volume = np.asarray(volume, dtype=np.float32)
    output = np.zeros(volume.shape, dtype=np.uint8)
    for index, image in enumerate(volume):
        finite = np.isfinite(image)
        if not finite.any():
            continue
        lo = float(image[finite].min())
        hi = float(image[finite].max())
        if hi <= lo:
            continue
        normalized = np.zeros_like(image, dtype=np.float32)
        normalized[finite] = np.clip((image[finite] - lo) / (hi - lo), 0.0, 1.0)
        output[index] = np.rint(normalized * 255.0).astype(np.uint8)
    return output


def atomic_save(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if not args.input_h5.is_file():
        raise FileNotFoundError(args.input_h5)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input_h5, "r") as source:
        patient_ids = sorted(source.keys(), key=natural_key)
        for patient_id in tqdm(patient_ids, unit="patient"):
            output_path = args.output_dir / f"{patient_id}.npy"
            expected_shape = source[patient_id][args.image_key].shape[:3]
            if output_path.is_file() and not args.overwrite:
                existing = np.load(output_path, mmap_mode="r", allow_pickle=False)
                if existing.shape == expected_shape and existing.dtype == np.uint8:
                    continue
                raise ValueError(f"Invalid existing staged input: {output_path}")
            atomic_save(
                output_path,
                normalize_slices(source[patient_id][args.image_key][...]),
            )

    print(f"Exported {len(patient_ids)} T2 volumes to {args.output_dir}")


if __name__ == "__main__":
    main()
