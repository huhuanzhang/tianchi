"""Protect demonstration targets, temporal consistency, and evaluation inputs."""
from pathlib import Path
import copy
import json
import unittest

import torch

from .clean_augment import CAMERAS, GRIPPERS, JOINTS, CleanAugment, sample_seed

CONFIGS = Path(__file__).parent / "configs"


class AugmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.config = json.loads((CONFIGS / "conservative.json").read_text())
        self.config["color"]["probability"] = 1
        self.config["state"]["probability"] = 1
        self.stats = {"features": {"observation.state": {"std": [0.2]*14}}}
        image = torch.arange(3*12*16).reshape(3,12,16).remainder(256).to(torch.uint8)
        self.sample = {key: torch.stack([image.clone(), image.clone()]) for key in CAMERAS}
        self.sample.update({"observation.state": torch.linspace(-0.2,0.8,14),
                            "action": torch.randn(16,14), "action_is_pad": torch.zeros(16,dtype=torch.bool),
                            "task": "official existing instruction", "task_index": 42})
        self.augment = CleanAugment(self.config, self.stats)

    def test_targets_masks_text_and_source_are_unchanged(self):
        snapshot = copy.deepcopy(self.sample)
        changed = self.augment(self.sample, seed=100)
        for key in ("action", "action_is_pad", "task", "task_index"):
            self.assertIs(changed[key], self.sample[key])
        for key, value in snapshot.items():
            if isinstance(value,torch.Tensor):
                self.assertTrue(torch.equal(self.sample[key], value))
        self.assertFalse(torch.equal(changed[CAMERAS[0]], self.sample[CAMERAS[0]]))

    def test_same_parameters_for_all_views_and_times(self):
        changed = self.augment(self.sample, seed=101)
        for key in CAMERAS:
            self.assertTrue(torch.equal(changed[key],changed[CAMERAS[0]]))
            self.assertTrue(torch.equal(changed[key][0],changed[key][1]))

    def test_grippers_unchanged_and_joint_noise_bounded(self):
        for seed in range(30):
            changed = self.augment(self.sample, seed=seed)
            state = changed["observation.state"]
            original = self.sample["observation.state"]
            self.assertTrue(torch.equal(state[list(GRIPPERS)], original[list(GRIPPERS)]))
            delta = (state-original).abs()
            self.assertTrue(torch.all(delta <= self.augment.sigma*3 + 1e-7))
            self.assertTrue(torch.any(delta[list(JOINTS)] > 0))

    def test_constant_joints_receive_no_noise(self):
        self.stats["features"]["observation.state"]["std"][0] = 0
        augment = CleanAugment(self.config, self.stats)
        output = augment(self.sample,seed=123)
        self.assertEqual(output["observation.state"][0],self.sample["observation.state"][0])

    def test_state_offset_shared_across_time(self):
        self.sample["observation.state"] = torch.zeros(3,14)
        changed = self.augment(self.sample,seed=77)
        self.assertTrue(torch.equal(changed["observation.state"][0],changed["observation.state"][2]))

    def test_validation_bypass_is_exact(self):
        result = self.augment(self.sample,seed=500,training=False)
        for key in self.sample:
            self.assertIs(result[key],self.sample[key])

    def test_replay_and_epoch_variation(self):
        seed = sample_seed(532514,0,100)
        result = self.augment(self.sample,seed=seed)
        replay = self.augment(self.sample,seed=seed)
        different = self.augment(self.sample,seed=sample_seed(532514,1,100))
        for key in CAMERAS + ("observation.state",):
            self.assertTrue(torch.equal(result[key],replay[key]))
        self.assertFalse(torch.equal(result[CAMERAS[0]],different[CAMERAS[0]]))

    def test_uint8_and_float_images_keep_shape_type_and_range(self):
        for dtype in (torch.uint8,torch.float32):
            sample = dict(self.sample)
            for key in CAMERAS:
                sample[key] = self.sample[key] if dtype == torch.uint8 else self.sample[key].float()/255
            changed = self.augment(sample,seed=100)
            for key in CAMERAS:
                self.assertEqual(changed[key].shape,sample[key].shape)
                self.assertEqual(changed[key].dtype,dtype)
                self.assertGreaterEqual(changed[key].min().item(),0)
                self.assertLessEqual(changed[key].max().item(),255 if dtype == torch.uint8 else 1)

    def test_normalized_or_padded_inputs_are_rejected(self):
        sample = dict(self.sample)
        sample[CAMERAS[0]] = self.sample[CAMERAS[0]].float()/255 - 0.5
        with self.assertRaises(ValueError):
            self.augment(sample,seed=10)
        sample = dict(self.sample)
        sample["observation.state"] = torch.zeros(55)
        with self.assertRaises(ValueError):
            self.augment(sample,seed=10)

    def test_baseline_leaves_every_field_unchanged(self):
        baseline = CleanAugment(json.loads((CONFIGS / "baseline.json").read_text()),self.stats)
        output = baseline(self.sample,seed=99)
        for key in self.sample:
            self.assertIs(output[key],self.sample[key])


if __name__ == "__main__":
    unittest.main()
