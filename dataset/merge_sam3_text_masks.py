#!/usr/bin/env python3
"""Merge validated staged Medical-SAM3 and MedSAM3 masks into the source H5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


DEFAULT_H5 = Path(__file__).with_name("patients_dataset_postmask.h5")
DEFAULT_MEDICAL = Path("/tmp/pcahsd_medicalsam3_masks")
DEFAULT_MEDSAM = Path("/tmp/pcahsd_medsam3_masks")
SOURCES = {
    "medicalsam3": DEFAULT_MEDICAL,
    "medsam3": DEFAULT_MEDSAM,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--medicalsam3-dir", type=Path, default=DEFAULT_MEDICAL)
    parser.add_argument("--medsam3-dir", type=Path, default=DEFAULT_MEDSAM)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def natural_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def load_metadata(directory: Path) -> dict:
    with (directory / "metadata.json").open() as stream:
        return json.load(stream)


def validate_all(h5_path: Path, directories: dict[str, Path]) -> None:
    with h5py.File(h5_path, "r") as source:
        patient_ids = sorted(source.keys(), key=natural_key)
        for key, directory in directories.items():
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            metadata = load_metadata(directory)
            if metadata.get("prompt") != "prostate":
                raise ValueError(f"{key}: expected prompt 'prostate'")
            for patient_id in patient_ids:
                path = directory / f"{patient_id}.npy"
                if not path.is_file():
                    raise FileNotFoundError(path)
                mask = np.load(path, mmap_mode="r", allow_pickle=False)
                expected_shape = source[patient_id]["T2"].shape[:3]
                if mask.shape != expected_shape or mask.dtype != np.uint8:
                    raise ValueError(
                        f"{path}: got {mask.dtype} {mask.shape}; "
                        f"expected uint8 {expected_shape}"
                    )
                unique = np.unique(mask)
                if not np.all(np.isin(unique, (0, 1))):
                    raise ValueError(f"{path}: non-binary values {unique}")


def main() -> None:
    args = parse_args()
    directories = {
        "medicalsam3": args.medicalsam3_dir,
        "medsam3": args.medsam3_dir,
    }
    validate_all(args.h5, directories)
    metadata = {key: load_metadata(path) for key, path in directories.items()}

    with h5py.File(args.h5, "r+") as target:
        patient_ids = sorted(target.keys(), key=natural_key)
        for patient_id in tqdm(patient_ids, unit="patient"):
            group = target[patient_id]
            for key, directory in directories.items():
                if key in group and not args.overwrite:
                    continue
                mask = np.load(directory / f"{patient_id}.npy", allow_pickle=False)
                data = mask[..., None]
                temporary_key = f"_tmp_{key}"
                if temporary_key in group:
                    del group[temporary_key]
                dataset = group.create_dataset(
                    temporary_key,
                    data=data,
                    dtype=np.uint8,
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                    chunks=(1, data.shape[1], data.shape[2], 1),
                )
                model_metadata = metadata[key]
                dataset.attrs["model"] = key
                dataset.attrs["image_key"] = "T2"
                dataset.attrs["text_prompt"] = "prostate"
                dataset.attrs["confidence_threshold"] = model_metadata["threshold"]
                dataset.attrs["checkpoint"] = model_metadata["checkpoint"]
                if key in group:
                    del group[key]
                group.move(temporary_key, key)
            target.flush()

    print(f"Merged keys {list(directories)} into {args.h5}")


if __name__ == "__main__":
    main()
