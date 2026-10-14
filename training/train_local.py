"""Train the native LingBot action expert on augmented clean demonstrations."""
from pathlib import Path
import argparse
import json
import random
import shutil
import statistics
import time

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Subset

from .native_model import ROOT, REPO, load_model, is_trainable
from .clean_dataset import CleanDataset
from augmentation.prepare import file_hash
from .rtc import assert_resume_compatible, export_rtc_assets


def gpu_batch(sample):
    batch = {}
    for key, value in sample.items():
        if (key in ("sample_index", "color_augmented", "state_augmented")
                or key.startswith(('joint_dropout_', 'image_crop_'))):
            continue
        dtype = torch.bfloat16 if key == "images" else torch.float32
        batch[key] = value.to("cuda", dtype=dtype if value.is_floating_point() else value.dtype)
    return batch


def export_runtime_assets(full, settings, model_config=None):
    metadata = ROOT/settings["vlm_config"]
    for name in ("tokenizer.json","tokenizer_config.json","vocab.json","merges.txt",
                 "special_tokens_map.json","added_tokens.json","chat_template.json",
                 "preprocessor_config.json","video_preprocessor_config.json"):
        if (metadata/name).is_file():
            shutil.copyfile(metadata/name,full/name)
    shutil.copyfile(ROOT/settings["norm_stats"],full/"clean_train_norm.json")
    robot = yaml.safe_load((REPO/"configs/robot_configs/robotwin.yaml").read_text())
    robot["norm_stats"] = str((full/"clean_train_norm.json").resolve())
    (full/"robotwin_clean.yaml").write_text(yaml.safe_dump(robot,sort_keys=False))
    if model_config is not None:
        export_rtc_assets(full, model_config)


def save_checkpoint(model, optimizer, output, settings, step, epoch, save_optimizer, samples_consumed=0):
    output.mkdir(parents=True, exist_ok=True)
    # CPU copies protect against mutating live parameters during serialization.
    parameters = {name: value.detach().cpu().contiguous() for name, value in model.named_parameters() if is_trainable(name)}
    path = output / "trainable.safetensors"
    save_file(parameters, path, metadata={"base_model": settings["base_model"], "training_source": "official_clean_only"})
    probe = "model.action_out_proj.weight"
    with safe_open(path, framework="pt", device="cpu") as saved:
        if not torch.equal(saved.get_tensor(probe), parameters[probe]):
            raise ValueError("Saved parameter verification failed")
    del parameters
    model.config.save_pretrained(output / "config")
    if save_optimizer:
        torch.save(optimizer.state_dict(), output / "optimizer.pt")
    torch.save({"cpu":torch.get_rng_state(), "cuda":torch.cuda.get_rng_state_all()},output/"rng_state.pt")
    metadata = {"step": step, "epoch": epoch, "settings": settings,
                "samples_consumed_in_epoch":samples_consumed,
                "base_manifest_sha256": file_hash(ROOT / "manifests/download_manifest.json"),
                "split_sha256": file_hash(ROOT / settings["split"]),
                "augment_config_sha256": file_hash(ROOT / settings["augment_config"]),
                "native_norm_sha256": file_hash(ROOT / settings["norm_stats"]),
                "checkpoint_reload_probe_passed": True}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return str(path)


def validate(model, dataset, count, seed):
    if not len(dataset):
        return None
    indices = random.Random(seed).sample(range(len(dataset)), min(count, len(dataset)))
    loader = DataLoader(Subset(dataset, indices), batch_size=1, num_workers=0)
    model.eval()
    losses = []
    with torch.no_grad():
        for sample in loader:
            if (sample["color_augmented"].any() or sample["state_augmented"].any()
                    or sample['joint_dropout_mask'].any() or sample['image_crop_mask'].any()):
                raise ValueError("Validation augmentation must be disabled")
            options = {'rtc_delay': 0} if getattr(model.config, 'rtc_training', {}).get('enabled', False) else {}
            losses.append(model(**gpu_batch(sample), **options)[0].item())
    model.train()
    return statistics.mean(losses)

