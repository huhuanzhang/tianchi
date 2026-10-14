"""Augment raw LeRobot inputs on CPU; never edit demonstration targets."""
from pathlib import Path
from typing import Mapping
import hashlib
import json

import torch

from .image_crop import RandomCameraCrop

CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
JOINTS = tuple(range(6)) + tuple(range(7, 13))
GRIPPERS = (6, 13)


def sample_seed(base_seed: int, epoch: int, frame_index: int, rank: int = 0) -> int:
    """Replay a sample independently of worker order; vary it every epoch/rank."""
    key = f"{base_seed}:{epoch}:{frame_index}:{rank}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little") % (2**63)


class CleanAugment:
    """Apply replayed camera crops and shared color transforms to raw inputs.

    Images: CPU RGB tensors, (C,H,W) or (T,C,H,W), uint8 or float in [0,1].
    State: CPU floating point tensor (...,14), in raw joint-angle units.
    A supplied seed makes augmentation reproducible across DataLoader workers.
    Call before image/state normalization, padding, or sin/cos conversion.
    """

    def __init__(self, config: Mapping, state_stats: Mapping):
        self.config = json.loads(json.dumps(config))
        if self.config.get("instruction_replacement", False):
            raise ValueError("Instruction replacement requires verified task/seen-pool mapping; this module leaves text unchanged")
        self.image_keys = tuple(self.config.get("image_keys", CAMERAS))
        if not self.image_keys or len(set(self.image_keys)) != len(self.image_keys):
            raise ValueError("image_keys must be a nonempty list of distinct keys")
        self.color = self.config["color"]
        self.state = self.config["state"]
        crop = self.config.get('crop', {})
        self.crop = RandomCameraCrop(crop.get('fraction', 1.0),
                                     crop.get('camera_probability', 0.0))
        for settings in (self.color, self.state):
            if not 0 <= settings["probability"] <= 1:
                raise ValueError("Augmentation probabilities must be in [0,1]")
        for name in ("brightness", "contrast", "saturation"):
            low, high = self.color[name]
            if not 0 < low <= high:
                raise ValueError(f"Invalid {name} factor range")
        for name in ("std_fraction", "max_sigma_radians", "clip_sigma"):
            if self.state[name] < 0:
                raise ValueError(f"Negative state noise setting: {name}")
        observed_std = torch.tensor(
            state_stats["features"]["observation.state"]["std"], dtype=torch.float32
        )
        if observed_std.shape != (14,) or not torch.isfinite(observed_std).all() or (observed_std < 0).any():
            raise ValueError("Expected 14 finite, nonnegative raw-state standard deviations")
        self.sigma = torch.minimum(
            observed_std * self.state["std_fraction"],
            torch.full((14,), self.state["max_sigma_radians"]),
        )
        self.sigma[list(GRIPPERS)] = 0

    @classmethod
    def from_files(cls, config_path, stats_path):
        return cls(json.loads(Path(config_path).read_text()), json.loads(Path(stats_path).read_text()))

    def __call__(self, sample: Mapping, *, seed: int, training: bool = True) -> dict:
        return self.apply_with_trace(sample, seed=seed, training=training)[0]

    @staticmethod
    def _image(image: torch.Tensor, parameters: dict) -> torch.Tensor:
        if not isinstance(image, torch.Tensor) or image.device.type != "cpu":
            raise ValueError("Augment decoded CPU tensors inside Dataset.__getitem__")
        if image.ndim not in (3, 4) or image.shape[-3] != 3:
            raise ValueError("Expected RGB (C,H,W) or (T,C,H,W); do not pass batches")
        if image.dtype != torch.uint8 and not image.is_floating_point():
            raise ValueError("Images must be uint8 or floating point RGB")
        pixels = image.to(torch.float32)
        if image.dtype == torch.uint8:
            pixels = pixels / 255
        elif not torch.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
            raise ValueError("Floating images must be unnormalized RGB in [0,1]")
        # Fixed contrast pivot: identical transform across cameras and timestamps.
        pixels = (pixels * parameters["brightness"] - 0.5) * parameters["contrast"] + 0.5
        weights = torch.tensor([0.299, 0.587, 0.114]).reshape(3, 1, 1)
        gray = (pixels * weights).sum(dim=-3, keepdim=True)
        pixels = (gray + (pixels - gray) * parameters["saturation"]).clamp(0, 1)
        if image.dtype == torch.uint8:
            return (pixels * 255).round().to(torch.uint8)
        return pixels.to(image.dtype)

    def apply_with_trace(self, sample: Mapping, *, seed: int, training: bool = True):
        output = dict(sample)
        trace = {"seed": seed, "training": training, "color_applied": False, "state_applied": False,
                 'crop_applied': False, 'crop_views': {}}
        if not training:
            return output, trace
        # Independent RNG stream preserves color, state noise and joint dropout
        # draws when only the camera crop configuration changes.
        output, crop_trace = self.crop(output, self.image_keys,
                                       seed=sample_seed(seed, 0, 1), training=training)
        trace.update(crop_applied=crop_trace['applied'], crop_views=crop_trace['views'])
        generator = torch.Generator(device="cpu").manual_seed(seed)

        def uniform(low, high):
            return low + (high - low) * torch.rand((), generator=generator).item()

        if uniform(0, 1) < self.color["probability"]:
            parameters = {name: uniform(*self.color[name]) for name in ("brightness", "contrast", "saturation")}
            for key in self.image_keys:
                if key not in sample:
                    raise KeyError(f"Missing configured image key: {key}")
                output[key] = self._image(output[key], parameters)
            trace.update(color_applied=True, color_parameters=parameters)
        if uniform(0, 1) < self.state["probability"]:
            original = sample["observation.state"]
            if not isinstance(original, torch.Tensor) or original.device.type != "cpu" or not original.is_floating_point():
                raise ValueError("State must be a floating point CPU tensor in raw angle units")
            if original.ndim < 1 or original.shape[-1] != 14 or not torch.isfinite(original).all():
                raise ValueError("Expected finite raw (...,14) state; augment before normalization/padding")
            # One sensor offset per clip: no artificial temporal jitter.
            noise = torch.randn(14, generator=generator).clamp(
                -self.state["clip_sigma"], self.state["clip_sigma"]
            ) * self.sigma
            perturbed = original + noise.to(original.dtype)
            # Keep the two gripper values bit-identical, including signed zeros.
            perturbed[..., list(GRIPPERS)] = original[..., list(GRIPPERS)]
            output["observation.state"] = perturbed
            trace.update(state_applied=True, state_sigma_radians=self.sigma.tolist(), state_offset=noise.tolist())
        # action, action masks, task_index and text are carried over verbatim.
        return output, trace
