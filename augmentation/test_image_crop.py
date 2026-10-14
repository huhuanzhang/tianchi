"""Protect crop geometry, camera selection, temporal replay and targets."""
import copy
import json
from pathlib import Path
import unittest

import torch

from .clean_augment import CAMERAS, CleanAugment, sample_seed
from .image_crop import RandomCameraCrop


class CameraCropTests(unittest.TestCase):
    def setUp(self):
        image = torch.arange(3*240*320).reshape(3, 240, 320).remainder(256).to(torch.uint8)
        self.sample = {key: torch.stack([image, image]) for key in CAMERAS}
        self.sample.update(action=torch.arange(14).float(),
                           **{'observation.state': torch.arange(14).float(), 'task': 'clean instruction'})

    def test_five_percent_is_per_axis_and_pixel_coordinates_are_exact(self):
        original = copy.deepcopy(self.sample)
        output, trace = RandomCameraCrop(.95, 1)(self.sample, CAMERAS, seed=1)
        self.assertTrue(trace['applied'])
        for key, box in trace['views'].items():
            self.assertEqual((box['height'], box['width']), (228, 304))
            self.assertLessEqual(box['top'], 12)
            self.assertLessEqual(box['left'], 16)
            expected = self.sample[key][..., box['top']:box['top']+228,
                                        box['left']:box['left']+304]
            self.assertTrue(torch.equal(output[key], expected))
            self.assertTrue(torch.equal(self.sample[key], original[key]))
        for key in ('action', 'observation.state', 'task'):
            self.assertIs(output[key], self.sample[key])

    def test_selected_views_and_temporal_frames_share_one_crop(self):
        output, trace = RandomCameraCrop(.95, 1)(self.sample, CAMERAS, seed=2)
        boxes = list(trace['views'].values())
        self.assertTrue(all(box == boxes[0] for box in boxes))
        for key in CAMERAS:
            self.assertTrue(torch.equal(output[key][0], output[key][1]))
            self.assertTrue(torch.equal(output[key], output[CAMERAS[0]]))

    def test_probability_selects_each_view_independently(self):
        small = {key: value[..., :20, :24] for key, value in self.sample.items()
                 if key in CAMERAS}
        counts = torch.zeros(3)
        partial = False
        for seed in range(1000):
            output, trace = RandomCameraCrop()(small, CAMERAS, seed=seed)
            for index, key in enumerate(CAMERAS):
                counts[index] += key in trace['views']
                if key not in trace['views']:
                    self.assertIs(output[key], small[key])
            partial |= 0 < len(trace['views']) < 3
        self.assertTrue(partial)
        self.assertTrue(torch.all((counts/1000 > .45) & (counts/1000 < .55)))

    def test_disabled_and_evaluation_paths_are_exact(self):
        for crop, training in ((RandomCameraCrop(.95, 0), True),
                               (RandomCameraCrop(1, 1), True),
                               (RandomCameraCrop(.95, 1), False)):
            output, trace = crop(self.sample, CAMERAS, seed=3, training=training)
            self.assertFalse(trace['applied'])
            self.assertEqual(trace['views'], {})
            for key in self.sample:
                self.assertIs(output[key], self.sample[key])

    def test_replay_and_local_rng_are_independent_of_training_rng(self):
        crop = RandomCameraCrop(.95, 1)
        rng = torch.get_rng_state().clone()
        _, first = crop(self.sample, CAMERAS, seed=sample_seed(532514, 0, 1))
        _, replay = crop(self.sample, CAMERAS, seed=sample_seed(532514, 0, 1))
        _, next_epoch = crop(self.sample, CAMERAS, seed=sample_seed(532514, 1, 1))
        self.assertEqual(first, replay)
        self.assertNotEqual(first, next_epoch)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_float_images_mixed_resolution_and_tiny_images(self):
        sample = {CAMERAS[0]: self.sample[CAMERAS[0]].float()/255,
                  CAMERAS[1]: self.sample[CAMERAS[1]][..., :200, :300],
                  CAMERAS[2]: torch.ones(3, 1, 1, dtype=torch.uint8)}
        output, trace = RandomCameraCrop(.95, 1)(sample, CAMERAS, seed=4)
        for key in CAMERAS[:2]:
            box = trace['views'][key]
            h, w = sample[key].shape[-2:]
            self.assertEqual(output[key].shape[-2:], (int(h*.95), int(w*.95)))
            self.assertEqual(output[key].dtype, sample[key].dtype)
            self.assertLessEqual(box['top']+box['height'], h)
            self.assertLessEqual(box['left']+box['width'], w)
        self.assertIs(output[CAMERAS[2]], sample[CAMERAS[2]])

    def test_invalid_configuration_and_normalized_inputs_are_rejected(self):
        for fraction in (0, -.1, 1.1, float('nan'), True, '.95'):
            with self.assertRaises(ValueError):
                RandomCameraCrop(fraction, 1)
        for probability in (-.1, 1.1, float('inf'), True, '.5'):
            with self.assertRaises(ValueError):
                RandomCameraCrop(.95, probability)
        sample = dict(self.sample)
        sample[CAMERAS[0]] = sample[CAMERAS[0]].float()/255-.5
        with self.assertRaises(ValueError):
            RandomCameraCrop(.95, 1)(sample, CAMERAS, seed=5)

    def test_raw_pipeline_preserves_color_rng_and_all_nonimage_fields(self):
        config = json.loads((Path(__file__).parent/'configs/color_crop_joint_dropout.json').read_text())
        config['crop']['camera_probability'] = 1
        stats = {'features': {'observation.state': {'std': [.2]*14}}}
        no_crop = copy.deepcopy(config)
        no_crop['crop']['camera_probability'] = 0
        output, trace = CleanAugment(config, stats).apply_with_trace(self.sample, seed=20)
        _, baseline_trace = CleanAugment(no_crop, stats).apply_with_trace(self.sample, seed=20)
        self.assertTrue(trace['crop_applied'])
        self.assertEqual(trace['color_applied'], baseline_trace['color_applied'])
        self.assertEqual(trace.get('color_parameters'), baseline_trace.get('color_parameters'))
        for key in ('action', 'observation.state', 'task'):
            self.assertIs(output[key], self.sample[key])
        evaluation, trace = CleanAugment(config, stats).apply_with_trace(
            self.sample, seed=20, training=False)
        self.assertFalse(trace['crop_applied'])
        for key in self.sample:
            self.assertIs(evaluation[key], self.sample[key])


if __name__ == '__main__':
    unittest.main()
