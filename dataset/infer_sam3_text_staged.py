#!/usr/bin/env python3
"""Infer prostate masks from staged T2 volumes with Medical-SAM3 or MedSAM3."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


MEDICALSAM3_ROOT = Path("/data/users/lly/projects/Medical-SAM3")
MEDSAM3_ROOT = Path("/data/users/lly/projects/MedSAM3")
DEFAULT_INPUT = Path("/tmp/pcahsd_t2_sam3_input")
DEFAULT_MEDICAL_CHECKPOINT = MEDICALSAM3_ROOT / "assets/checkpoint_2D.pt"
DEFAULT_BASE_CHECKPOINT = Path("/data/users/lly/sam3.pt")
DEFAULT_LORA_WEIGHTS = MEDSAM3_ROOT / "outputs/medsam3_v1/best_lora_weights.pt"
DEFAULT_LORA_CONFIG = MEDSAM3_ROOT / "configs/full_lora_config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("medicalsam3", "medsam3"), required=True)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", default="prostate")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--medical-checkpoint", type=Path, default=DEFAULT_MEDICAL_CHECKPOINT)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--lora-weights", type=Path, default=DEFAULT_LORA_WEIGHTS)
    parser.add_argument("--lora-config", type=Path, default=DEFAULT_LORA_CONFIG)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def natural_key(path: Path) -> tuple[int, int | str]:
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.stem)


def build_medicalsam3(args: argparse.Namespace):
    sys.path.insert(0, str(MEDICALSAM3_ROOT))
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    threshold = 0.1 if args.threshold is None else args.threshold
    model = build_sam3_image_model(
        bpe_path=str(MEDICALSAM3_ROOT / "assets/bpe_simple_vocab_16e6.txt.gz"),
        device="cuda",
        checkpoint_path=str(args.medical_checkpoint),
        load_from_HF=False,
        eval_mode=True,
        enable_segmentation=True,
    )
    processor = Sam3Processor(
        model,
        resolution=1008,
        device="cuda",
        confidence_threshold=threshold,
    )
    return model, processor, threshold, str(args.medical_checkpoint)


def build_medsam3(args: argparse.Namespace):
    sys.path.insert(0, str(MEDSAM3_ROOT))
    import yaml
    from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    threshold = 0.5 if args.threshold is None else args.threshold
    with args.lora_config.open() as stream:
        config = yaml.safe_load(stream)

    model = build_sam3_image_model(
        bpe_path=str(MEDSAM3_ROOT / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
        device="cuda",
        checkpoint_path=str(args.base_checkpoint),
        load_from_HF=False,
        eval_mode=True,
        enable_segmentation=True,
    )
    lora = config["lora"]
    model = apply_lora_to_model(
        model,
        LoRAConfig(
            rank=lora["rank"],
            alpha=lora["alpha"],
            dropout=0.0,
            target_modules=lora["target_modules"],
            apply_to_vision_encoder=lora["apply_to_vision_encoder"],
            apply_to_text_encoder=lora["apply_to_text_encoder"],
            apply_to_geometry_encoder=lora["apply_to_geometry_encoder"],
            apply_to_detr_encoder=lora["apply_to_detr_encoder"],
            apply_to_detr_decoder=lora["apply_to_detr_decoder"],
            apply_to_mask_decoder=lora["apply_to_mask_decoder"],
        ),
    )
    load_lora_weights(model, str(args.lora_weights))
    model.to("cuda").eval()
    processor = Sam3Processor(
        model,
        resolution=1008,
        device="cuda",
        confidence_threshold=threshold,
    )
    checkpoint_description = (
        f"base={args.base_checkpoint};lora={args.lora_weights};"
        f"config={args.lora_config}"
    )
    return model, processor, threshold, checkpoint_description


def best_mask(state: dict, height: int, width: int) -> tuple[np.ndarray, float]:
    masks = state.get("masks")
    scores = state.get("scores")
    if masks is None or scores is None or len(masks) == 0 or len(scores) == 0:
        return np.zeros((height, width), dtype=np.uint8), float("nan")

    best_index = int(scores.argmax().item())
    score = float(scores[best_index].item())
    mask = masks[best_index].detach().cpu().numpy().squeeze().astype(np.uint8)
    if mask.shape != (height, width):
        mask_image = Image.fromarray(mask * 255, mode="L")
        mask = np.asarray(
            mask_image.resize((width, height), Image.Resampling.NEAREST)
        )
        mask = (mask > 127).astype(np.uint8)
    else:
        mask = (mask > 0).astype(np.uint8)
    return mask, score


def atomic_save(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    os.replace(temporary, path)


def infer_volume(processor, volume: np.ndarray, prompt: str) -> tuple[np.ndarray, list[float]]:
    if volume.ndim != 3 or volume.dtype != np.uint8:
        raise ValueError(f"Expected uint8 (slices,H,W), got {volume.dtype} {volume.shape}")
    masks = np.zeros(volume.shape, dtype=np.uint8)
    scores: list[float] = []
    for slice_index, image in enumerate(volume):
        rgb = np.repeat(image[..., None], 3, axis=-1)
        state = processor.set_image(Image.fromarray(rgb, mode="RGB"))
        state = processor.set_text_prompt(state=state, prompt=prompt)
        masks[slice_index], score = best_mask(state, image.shape[0], image.shape[1])
        scores.append(score)
    return masks, scores


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this batch inference.")
    if not args.input_dir.is_dir():
        raise FileNotFoundError(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    print(
        f"model={args.model} visible_gpu={torch.cuda.get_device_name(0)} "
        f"prompt={args.prompt!r}"
    )

    if args.model == "medicalsam3":
        _, processor, threshold, checkpoint = build_medicalsam3(args)
    else:
        _, processor, threshold, checkpoint = build_medsam3(args)

    input_paths = sorted(args.input_dir.glob("*.npy"), key=natural_key)
    if args.max_patients is not None:
        input_paths = input_paths[: args.max_patients]

    score_summary: dict[str, dict[str, float | int]] = {}
    started = time.perf_counter()
    for input_path in tqdm(input_paths, unit="patient"):
        output_path = args.output_dir / input_path.name
        volume = np.load(input_path, allow_pickle=False)
        if output_path.is_file() and not args.overwrite:
            existing = np.load(output_path, mmap_mode="r", allow_pickle=False)
            if existing.shape == volume.shape and existing.dtype == np.uint8:
                continue
            raise ValueError(f"Invalid existing output: {output_path}")

        masks, scores = infer_volume(processor, volume, args.prompt)
        atomic_save(output_path, masks)
        finite_scores = np.asarray(scores, dtype=np.float32)
        finite_scores = finite_scores[np.isfinite(finite_scores)]
        score_summary[input_path.stem] = {
            "nonempty_slices": int(np.count_nonzero(masks.reshape(masks.shape[0], -1).sum(1))),
            "foreground_pixels": int(masks.sum()),
            "mean_best_score": float(finite_scores.mean()) if finite_scores.size else float("nan"),
        }

    elapsed = time.perf_counter() - started
    metadata = {
        "model": args.model,
        "prompt": args.prompt,
        "threshold": threshold,
        "checkpoint": checkpoint,
        "selected_patients": len(input_paths),
        "elapsed_seconds": elapsed,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "scores": score_summary,
    }
    with (args.output_dir / "metadata.json").open("w") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=True)
    print(f"Finished {len(input_paths)} patients in {elapsed:.1f}s: {args.output_dir}")


if __name__ == "__main__":
    main()
