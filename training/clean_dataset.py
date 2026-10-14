"""Use native feature mapping on immutable clean frames with online augmentation."""
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import bisect
import json
import subprocess

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize
from transformers import AutoImageProcessor, AutoTokenizer

from augmentation import CleanAugment, sample_seed
from augmentation.clean_augment import CAMERAS
from augmentation.joint_dropout import JointValueDropout
from augmentation.prepare import file_hash
from .native_model import ROOT, REPO


class ImageProcessor:
    def __init__(self, path, size):
        self.size = size
        self.processor = AutoImageProcessor.from_pretrained(path, use_fast=False)

    def __call__(self, image):
        image = resize(image, [self.size, self.size], antialias=True)
        return self.processor(images=image, do_resize=False, return_tensors="pt")


class CleanDataset(Dataset):
    def __init__(self, settings, model_config, training=True):
        from lingbotvla.data.vla_data.utils import FeatureTransform
        self.settings = settings
        self.training = training
        self.epoch = 0
        split_path = ROOT / settings["split"]
        self.split = json.loads(split_path.read_text())
        stats_path = ROOT / settings["augment_stats"]
        norm_path = ROOT / settings["norm_stats"]
        for path in (stats_path, norm_path):
            if json.loads(path.read_text())["split_sha256"] != file_hash(split_path):
                raise ValueError(f"Statistics and split mismatch: {path}")
        self.augment = CleanAugment.from_files(ROOT / settings["augment_config"], stats_path)
        # Missing keys preserve the behavior of historical augmentation files.
        self.joint_dropout = JointValueDropout(
            self.augment.config.get('joint_dropout_probability', 0.0))
        self.root = ROOT / settings.get("dataset_root", self.split["dataset_path"])
        self.rank = settings.get("rank", 0)
        self.info = json.loads((self.root / "meta/info.json").read_text())
        self.episodes = {e["episode_index"]: e for e in map(json.loads, (self.root / "meta/episodes.jsonl").read_text().splitlines())}
        self.tasks = {t["task_index"]: t["task"] for t in map(json.loads, (self.root / "meta/tasks.jsonl").read_text().splitlines())}
        self.ids = self.split["train_episodes" if training else "validation_episodes"]
        self.ends = np.cumsum([self.episodes[e]["length"] for e in self.ids]).tolist()
        processor = SimpleNamespace(
            tokenizer=AutoTokenizer.from_pretrained(ROOT / settings["vlm_config"]),
            image_processor=ImageProcessor(ROOT / settings["vlm_config"], settings["image_size"]),
        )
        data = SimpleNamespace(
            joints=["{'arm.position': 14}", "{'end.position': 14}", "{'effector.position': 2}"],
            cameras=["camera_top", "camera_wrist_left", "camera_wrist_right"],
            norm_type=["{'arm.position': 'bounds_99_woclip'}", "{'end.position': 'bounds_99_woclip'}", "{'effector.position': 'bounds_99_woclip'}"],
        )
        self.transform = FeatureTransform(
            str(REPO / "configs/robot_configs/robotwin.yaml"), data, model_config, processor,
            chunk_size=settings["chunk_size"], norm_stats_path=str(norm_path),
            use_depth_align=False, image_augment=False, use_future_image=False,
        )
        self.arm_state_mask = torch.zeros(model_config.max_state_dim, dtype=torch.bool)
        offset = 0
        for joint in self.transform.feature_config.joints:
            width = self.transform.feature_config.joints_max_dim[joint]
            if joint == 'arm.position':
                self.arm_state_mask[offset:offset+width] = True
            offset += width
        self.cache = OrderedDict()

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def _episode(self, idx):
        if idx in self.cache:
            self.cache.move_to_end(idx)
            return self.cache[idx]
        substitutions = {"episode_index": idx, "episode_chunk": idx // self.info["chunks_size"]}
        table = pq.read_table(self.root / self.info["data_path"].format(**substitutions), use_threads=False)
        values = {key: np.asarray(table[key].to_pylist()) for key in table.column_names}
        length = len(values["index"])
        videos = {}
        for key in CAMERAS:
            path = self.root / self.info["video_path"].format(video_key=key, **substitutions)
            decoded = subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-filter_threads", "1",
                 "-i", str(path), "-threads", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                capture_output=True, check=True,
            ).stdout
            frames = np.frombuffer(decoded, np.uint8)
            if frames.size != length * 240 * 320 * 3:
                raise ValueError(f"Frame count mismatch: {path}")
            videos[key] = frames.reshape(length, 240, 320, 3)
        self.cache[idx] = (values, videos)
        while len(self.cache) > self.settings.get("episode_cache_size", 2):
            self.cache.popitem(last=False)
        return values, videos

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        position = bisect.bisect_right(self.ends, index)
        episode = self.ids[position]
        offset = index - (self.ends[position-1] if position else 0)
        values, videos = self._episode(episode)
        length = len(values["index"])
        horizon = self.settings["chunk_size"]
        future = np.arange(offset, offset+horizon)
        padded = future >= length
        actions = values["action"][np.minimum(future, length-1)].copy()
        raw = {"observation.state": torch.tensor(values["observation.state"][offset], dtype=torch.float32),
               "action": torch.tensor(actions, dtype=torch.float32),
               "action_is_pad": torch.from_numpy(padded),
               "task": self.tasks[int(values["task_index"][offset])],
               "task_index": torch.tensor(int(values["task_index"][offset])),
               "episode_index": torch.tensor(episode), "frame_index": torch.tensor(offset),
               "timestamp": torch.tensor(float(values["timestamp"][offset])),
               "index": torch.tensor(int(values["index"][offset]))}
        for key in CAMERAS:
            raw[key] = torch.from_numpy(videos[key][offset].copy()).permute(2,0,1)
        seed = sample_seed(self.settings["seed"], self.epoch, int(raw["index"]), self.rank)
        augmented, trace = self.augment.apply_with_trace(raw, seed=seed, training=self.training)
        if not torch.equal(raw["action"], augmented["action"]):
            raise ValueError("Augmentation changed expert targets")
        item = self.transform.apply(augmented)
        eligible = self.arm_state_mask & item['state_joint_mask']
        item['state'], dropped = self.joint_dropout(
            item['state'], eligible, seed=sample_seed(seed, 0, 0), training=self.training)
        keys = ("images", "img_masks", "state", "lang_tokens", "lang_masks", "actions", "joint_mask", "action_is_pad", "image_grid_thw")
        result = {key: item[key] for key in keys}
        # The native feature mask covers real joint dimensions; also exclude
        # repeated end-of-episode frames that only pad the action horizon.
        result["joint_mask"] = result["joint_mask"] & ~result["action_is_pad"].unsqueeze(-1)
        result["sample_index"] = int(values["index"][offset])
        result["color_augmented"] = trace["color_applied"]
        result["state_augmented"] = trace["state_applied"]
        result['image_crop_mask'] = torch.tensor(
            [key in trace['crop_views'] for key in self.augment.image_keys], dtype=torch.bool)
        result['image_crop_count'] = int(result['image_crop_mask'].sum())
        result['image_crop_eligible_count'] = len(self.augment.image_keys)
        result['image_crop_boxes'] = torch.tensor([
            [trace['crop_views'][key][field] for field in ('top', 'left', 'height', 'width')]
            if key in trace['crop_views'] else [0, 0, 0, 0]
            for key in self.augment.image_keys], dtype=torch.int64)
        result['joint_dropout_mask'] = dropped
        result['joint_dropout_count'] = int(dropped.sum())
        result['joint_dropout_eligible_count'] = int(eligible.sum())
        return result
