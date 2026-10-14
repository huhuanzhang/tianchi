"""Verify joint masking, replay, temporal consistency and gradients."""
import unittest

import torch

from .joint_dropout import JointValueDropout


class JointDropoutTests(unittest.TestCase):
    def setUp(self):
        self.state = torch.linspace(-0.9, 0.9, 55)
        self.eligible = torch.zeros(55, dtype=torch.bool)
        self.eligible[:12] = True

    def test_endpoints_preserve_grippers_padding_and_source(self):
        original = self.state.clone()
        output, mask = JointValueDropout(1)(self.state, self.eligible, seed=10)
        self.assertTrue(torch.equal(mask, self.eligible))
        self.assertTrue(torch.equal(output[mask], torch.zeros(12)))
        self.assertTrue(torch.equal(output[~mask], original[~mask]))
        self.assertTrue(torch.equal(self.state, original))
        output, mask = JointValueDropout(0)(self.state, self.eligible, seed=10)
        self.assertIs(output, self.state)
        self.assertFalse(mask.any())

    def test_evaluation_bypass_is_exact(self):
        output, mask = JointValueDropout(1)(self.state, self.eligible, seed=11, training=False)
        self.assertIs(output, self.state)
        self.assertFalse(mask.any())

    def test_one_mask_is_shared_across_time_without_rescaling(self):
        state = self.state.repeat(3, 1)
        output, mask = JointValueDropout(.5)(state, self.eligible, seed=12)
        self.assertTrue(mask.any())
        self.assertTrue(torch.equal(output[0], output[2]))
        self.assertTrue(torch.equal(output[:, ~mask], state[:, ~mask]))

    def test_replay_uses_local_rng_without_mutating_training_rng(self):
        rng = torch.get_rng_state().clone()
        first, mask1 = JointValueDropout(.5)(self.state, self.eligible, seed=20)
        second, mask2 = JointValueDropout(.5)(self.state, self.eligible, seed=20)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(mask1, mask2))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        masks = [JointValueDropout(.5)(self.state, self.eligible, seed=i)[1] for i in range(5)]
        self.assertTrue(any(not torch.equal(masks[0], mask) for mask in masks[1:]))

    def test_default_rate_is_per_joint(self):
        dropout = JointValueDropout()
        self.assertEqual(dropout.probability, .2)
        masks = torch.stack([dropout(self.state, self.eligible, seed=i)[1] for i in range(1000)])
        rates = masks[:, self.eligible].float().mean(0)
        self.assertTrue(torch.all((rates > .15) & (rates < .25)))
        self.assertFalse(masks[:, ~self.eligible].any())
        self.assertTrue(torch.any((masks.sum(1) > 0) & (masks.sum(1) < 12)))

    def test_masked_inputs_have_zero_gradient_and_survivors_unit_gradient(self):
        state = self.state.clone().requires_grad_()
        output, mask = JointValueDropout(.5)(state, self.eligible, seed=23)
        output.sum().backward()
        self.assertTrue(torch.equal(state.grad[mask], torch.zeros_like(state.grad[mask])))
        self.assertTrue(torch.equal(state.grad[~mask], torch.ones_like(state.grad[~mask])))

    def test_invalid_probabilities_and_masks_are_rejected(self):
        for value in [-.1, 1.1, float('nan'), float('inf'), True, '0.2']:
            with self.assertRaises(ValueError):
                JointValueDropout(value)
        for state, mask in [(self.state.long(), self.eligible),
                            (self.state, self.eligible.float()),
                            (self.state, self.eligible[:14])]:
            with self.assertRaises(ValueError):
                JointValueDropout()(state, mask, seed=1)


if __name__ == '__main__':
    unittest.main()
