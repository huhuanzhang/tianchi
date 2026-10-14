"""Check portable asset paths and decoder support before server training."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess

import torch

from .native_model import ROOT, make_config
from .clean_dataset import CleanDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "training/configs/default.json")
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    base = ROOT / settings["base_model"]
    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    for shard in set(index.values()):
        if not (base / shard).is_file():
            raise FileNotFoundError(base / shard)
    if not args.cpu_only and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; mount GPUs with NVIDIA Container Toolkit")
    config = make_config(settings)
    training = CleanDataset(settings, config, training=True)
    validation = CleanDataset(settings, config, training=False)
    item = training[100]
    assert item["actions"].shape == (50,55) and item["joint_mask"].any()
    if len(validation):
        sample = validation[100]
        assert not sample["color_augmented"] and not sample["state_augmented"] and not sample['joint_dropout_mask'].any()
        assert not sample['image_crop_mask'].any()
    report = {"status": "passed", "training_frames": len(training), "validation_frames": len(validation),
              "rtc": config.rtc_training,
              "action_expert_dropout": config.action_expert_dropout,
              "joint_dropout_probability": training.joint_dropout.probability,
              "joint_dropout_scope": "per_arm_joint_after_normalization_training_only",
              "image_crop": dict(fraction=training.augment.crop.fraction,
                                 camera_probability=training.augment.crop.camera_probability,
                                 validation='disabled; full image'),
              "dataset_root": str(training.root), "base_tensors": len(index),
              "ffmpeg": subprocess.check_output(["ffmpeg","-version"],text=True).splitlines()[0],
              "packages": {name:importlib.metadata.version(name) for name in ["torch","torchvision","transformers","triton","pyarrow"]},
              "gpus": [{"name":torch.cuda.get_device_name(i),"memory_gib":torch.cuda.get_device_properties(i).total_memory / 2**30} for i in range(torch.cuda.device_count())]}
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
