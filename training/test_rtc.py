"""Analytical checks for clean-prefix times, masks, AdaNorm and hard sampling."""
import copy
from types import SimpleNamespace
import unittest

import torch

from .native_model import REPO  # Registers the pinned native source path.
from .rtc import normalize_config, training_inputs, sample_prefix, assert_resume_compatible
from .train_local import save_checkpoint
from safetensors.torch import load_file
from transformers import PretrainedConfig
import json
from pathlib import Path
import tempfile
from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla import (
    AdaRMSNorm, FixAdaRMSNorm, FlowMatching as NativeFlow,
)


class FakeFlow:
    def __init__(self):
        self.config = SimpleNamespace(n_action_steps=50, max_action_dim=55, num_steps=10,
                                      rtc_training=normalize_config({'enabled': True}))
        self.calls = []
        self.cache_calls = 0
        self.qwenvl_with_expert = SimpleNamespace(forward=self.cache)

    def sample_time(self, batch, device):
        return torch.full((batch,), .5, device=device)

    def cache(self, **kwargs):
        self.cache_calls += 1
        return None, object(), None

    def embed_prefix(self, *args, **kwargs):
        return torch.zeros(2, 3, 4), torch.ones(2, 3, dtype=torch.bool), torch.zeros(2, 3, dtype=torch.bool), torch.zeros(3, 2, 3, dtype=torch.long), None, None

    def predict_velocity(self, state, masks, cache, x, times, **kwargs):
        self.calls.append((x.clone(), times.clone()))
        return torch.ones_like(x)

    def sample_actions(self, **kwargs):
        return kwargs['noise'] + 7


class TinyRTCPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.action_out_proj = torch.nn.Linear(3, 2)
        self.config = PretrainedConfig(rtc_training=normalize_config({'enabled': True}))
        self.model.config = self.config
        self.model.sample_time = lambda n, device: torch.rand(n, device=device)

    def update(self, optimizer):
        actions = torch.randn(4, 50, 2)
        noise, times, mask, _ = training_inputs(self.model, actions, torch.ones_like(actions, dtype=torch.bool))
        if times.ndim == 1:
            times = times[:, None].expand(-1, 50)
        x = times[:, :, None]*noise+(1-times[:, :, None])*actions
        predicted = self.model.action_out_proj(torch.cat([x, times[:, :, None]], dim=-1))
        loss = ((predicted-(noise-actions)).abs()*mask).sum()/mask.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class RTCTests(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(42)
        actions = torch.randn(2, 50, 55)
        mask = torch.zeros_like(actions, dtype=torch.bool)
        mask[:, :, :14] = True
        mask[1, 3:] = False
        return FakeFlow(), actions, mask

    def test_prefix_clean_time_and_postfix_only_loss_with_padding(self):
        flow, actions, mask = self.fixture()
        original = mask.clone()
        noise, time, postfix, delays = training_inputs(flow, actions, mask, delay=torch.tensor([12, 12]))
        self.assertEqual(delays.tolist(), [12, 2])
        self.assertEqual(postfix.sum(dim=(1, 2)).tolist(), [38 * 14, 14])
        noisy = time[:, :, None] * noise + (1 - time[:, :, None]) * actions
        torch.testing.assert_close(noisy[0, :12], actions[0, :12], rtol=0, atol=0)
        torch.testing.assert_close(noisy[1, :2], actions[1, :2], rtol=0, atol=0)
        self.assertTrue(torch.equal(original, mask))
        # An arbitrarily bad prefix or padded prediction contributes no direct loss.
        error = torch.zeros_like(actions)
        error[0, :12] = 1e6
        error[1, :2] = 1e6
        error[1, 3:] = 1e6
        self.assertEqual((error * postfix).sum().item(), 0)
        self.assertTrue(torch.equal(time[0, 12:], torch.full((38,), .5)))

    def test_zero_delay_keeps_exact_native_scalar_time(self):
        flow, actions, mask = self.fixture()
        t = torch.tensor([.23, .82])
        noise = torch.randn_like(actions)
        n, times, valid, delays = training_inputs(flow, actions, mask, noise=noise, time=t, delay=0)
        self.assertIs(n, noise)
        self.assertIs(times, t)
        self.assertTrue(torch.equal(mask, valid))
        self.assertEqual(delays.tolist(), [0, 0])

    def test_short_chunks_leave_supervision_and_invalid_inputs_fail(self):
        flow, actions, mask = self.fixture()
        mask[0, 1:] = False
        _, _, valid, delays = training_inputs(flow, actions, mask, delay=12)
        self.assertEqual(delays.tolist(), [0, 2])
        self.assertEqual(valid.sum(dim=(1, 2)).tolist(), [14, 14])
        with self.assertRaises(ValueError):
            training_inputs(flow, actions, mask, delay=13)
        with self.assertRaises(ValueError):
            training_inputs(flow, actions, mask, delay=1.5)
        with self.assertRaises(ValueError):
            training_inputs(flow, actions, torch.zeros_like(mask))
        mask[1, 4] = True
        with self.assertRaises(ValueError):
            training_inputs(flow, actions, mask)

    def test_random_delay_and_flow_rng_can_be_replayed(self):
        flow, actions, mask = self.fixture()
        rng = torch.get_rng_state()
        a = training_inputs(flow, actions, mask)
        torch.set_rng_state(rng)
        b = training_inputs(flow, actions, mask)
        for x, y in zip(a, b):
            self.assertTrue(torch.equal(x, y))
        old = {'chunk_size': 50}
        with self.assertRaises(ValueError):
            assert_resume_compatible(old, dict(old, rtc={'enabled': True}))
        assert_resume_compatible(old, dict(old, rtc={'enabled': False}))

    def test_adanorm_per_token_equals_separate_scalar_conditions_and_gradients(self):
        for cls in [AdaRMSNorm, FixAdaRMSNorm]:
            torch.manual_seed(5)
            norm = cls(8, 8)
            torch.nn.init.normal_(norm.gamma.weight)
            torch.nn.init.normal_(norm.beta.weight)
            x = torch.randn(2, 51, 8, requires_grad=True)
            condition = torch.randn(2, 51, 8, requires_grad=True)
            result = norm(x, condition)
            expected = torch.cat([norm(x[:, i:i+1], condition[:, i]) for i in range(51)], dim=1)
            torch.testing.assert_close(result, expected)
            result.square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all() and torch.isfinite(condition.grad).all())
            scalar = condition.detach()[:, 0]
            torch.testing.assert_close(norm(x.detach(), scalar), norm(x.detach(), scalar[:, None].expand(-1, 51, -1)))

    def test_native_suffix_per_token_time_preserves_scalar_behavior(self):
        torch.manual_seed(7)
        flow = SimpleNamespace(training=False, config=SimpleNamespace(proj_width=8, action_fp32=False),
                               state_proj=torch.nn.Linear(6, 8), action_in_proj=torch.nn.Linear(5, 8),
                               action_time_mlp_in=torch.nn.Linear(16, 8), action_time_mlp_out=torch.nn.Linear(8, 8))
        state, actions, times = torch.randn(2, 6), torch.randn(2, 50, 5), torch.tensor([.4, .8])
        old = NativeFlow.embed_suffix(flow, state, actions, times)
        new = NativeFlow.embed_suffix(flow, state, actions, times[:, None].expand(-1, 50))
        torch.testing.assert_close(old[1], new[1], rtol=0, atol=0)
        torch.testing.assert_close(old[0][:, None].expand(-1, 51, -1), new[0], rtol=0, atol=0)
        prefix_times = times[:, None].expand(-1, 50).clone()
        prefix_times[:, :6] = 0
        conditioned = NativeFlow.embed_suffix(flow, state, actions, prefix_times)
        torch.testing.assert_close(conditioned[0][:, 0], old[0], rtol=0, atol=0)
        self.assertFalse(torch.equal(conditioned[0][:, 1:7], new[0][:, 1:7]))

    def test_sampler_clamps_every_step_and_final_prefix_without_vjp(self):
        flow = FakeFlow()
        inputs = dict(state=torch.zeros(2, 55), images=None, img_masks=None,
                      lang_tokens=None, lang_masks=None)
        previous, noise = torch.randn(2, 20, 55), torch.randn(2, 50, 55)
        originals = previous.clone(), noise.clone()
        mask = torch.zeros(55, dtype=torch.bool)
        mask[:14] = True
        result = sample_prefix(flow, inputs, previous, torch.tensor([4, 7]), mask, noise)
        self.assertEqual(flow.cache_calls, 1)
        self.assertEqual(len(flow.calls), 10)
        for x, times in flow.calls:
            for batch, delay in enumerate([4, 7]):
                torch.testing.assert_close(x[batch, :delay], previous[batch, :delay] * mask, rtol=0, atol=0)
                self.assertTrue((times[batch, :delay] == 0).all())
        for batch, delay in enumerate([4, 7]):
            torch.testing.assert_close(result[batch, :delay], previous[batch, :delay] * mask, rtol=0, atol=0)
            torch.testing.assert_close(result[batch, delay:], noise[batch, delay:] - 1, atol=1e-6, rtol=1e-6)
        self.assertTrue(torch.equal(previous, originals[0]) and torch.equal(noise, originals[1]))

    def test_zero_delay_sampler_calls_native_path(self):
        flow = FakeFlow()
        inputs = dict(state=torch.zeros(2, 55))
        noise = torch.randn(2, 50, 55)
        result = sample_prefix(flow, inputs, None, 0, None, noise)
        torch.testing.assert_close(result, noise + 7, rtol=0, atol=0)
        self.assertEqual(flow.cache_calls, 0)

    def test_rtc_checkpoint_restores_prefix_rng_optimizer_and_next_update(self):
        from .native_model import ROOT
        torch.manual_seed(77)
        policy = TinyRTCPolicy()
        optimizer = torch.optim.AdamW(policy.parameters(), lr=.01)
        settings = json.loads((ROOT/'training/configs/server_rtc.json').read_text())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            for key in ("split", "norm_stats"):
                fixture = path / (key+"_fixture.json")
                fixture.write_text("{}\n")
                settings[key] = str(fixture)
            policy.update(optimizer)
            save_checkpoint(policy, optimizer, path, settings, 1, 0, True, 4)
            policy.update(optimizer)
            expected = copy.deepcopy(policy.state_dict())
            expected_delay = policy.model._rtc_last_delays.clone()
            restored = TinyRTCPolicy()
            restored.load_state_dict(load_file(path/'trainable.safetensors'))
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.01)
            restored_optimizer.load_state_dict(torch.load(path/'optimizer.pt', weights_only=True))
            torch.set_rng_state(torch.load(path/'rng_state.pt', weights_only=True)['cpu'])
            restored.update(restored_optimizer)
            for name, value in restored.state_dict().items():
                self.assertTrue(torch.equal(value, expected[name]), name)
            self.assertTrue(torch.equal(restored.model._rtc_last_delays, expected_delay))
            self.assertTrue(PretrainedConfig.from_pretrained(path/'config').rtc_training['enabled'])


if __name__ == '__main__':
    unittest.main()
