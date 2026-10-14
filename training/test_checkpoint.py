"""Check that saved optimizer/RNG state reproduces the next update after restart."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import load_file
from transformers import PretrainedConfig

from .native_model import ROOT
from .train_local import save_checkpoint
from .train_server import retain_checkpoints


class TinyActionPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.action_out_proj = torch.nn.Linear(3, 2)
        self.config = PretrainedConfig()

    def forward(self, value):
        return self.model.action_out_proj(value)


def update(policy, optimizer):
    optimizer.zero_grad()
    values, targets = torch.randn(4,3), torch.randn(4,2)
    loss = (policy(values)-targets).square().mean()
    loss.backward()
    optimizer.step()


class CheckpointTests(unittest.TestCase):
    def test_optimizer_and_rng_resume_matches_next_update_and_retention(self):
        torch.manual_seed(532514)
        policy = TinyActionPolicy()
        optimizer = torch.optim.AdamW(policy.parameters(), lr=0.01)
        settings = json.loads((ROOT / "training/configs/server_holdout.json").read_text())
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            for key in ("split", "norm_stats"):
                fixture = output / (key+"_fixture.json")
                fixture.write_text("{}\n")
                settings[key] = str(fixture)
            checkpoint = output / "step_000001"
            update(policy, optimizer)
            save_checkpoint(policy, optimizer, checkpoint, settings, 1, 0, True, 64)
            update(policy, optimizer)
            expected = copy.deepcopy(policy.state_dict())
            moments = copy.deepcopy(optimizer.state_dict())
            restarted = TinyActionPolicy()
            restarted.load_state_dict(load_file(checkpoint / "trainable.safetensors"))
            restored_optimizer = torch.optim.AdamW(restarted.parameters(), lr=0.01)
            restored_optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", weights_only=True))
            rng = torch.load(checkpoint / "rng_state.pt", weights_only=True)
            torch.set_rng_state(rng["cpu"])
            update(restarted, restored_optimizer)
            for name, value in restarted.state_dict().items():
                self.assertTrue(torch.equal(value, expected[name]), name)
            for key, state in restored_optimizer.state_dict()["state"].items():
                for name, value in state.items():
                    self.assertTrue(torch.equal(value, moments["state"][key][name]))
            for step in (2,3):
                save_checkpoint(restarted, restored_optimizer, output / f"step_{step:06d}",
                                settings, step, 0, True, step * 64)
            retain_checkpoints(output, 1)
            self.assertEqual([path.name for path in output.glob("step_*")],["step_000003"])


if __name__ == "__main__":
    unittest.main()
