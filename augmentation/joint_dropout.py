"""Drop normalized arm-joint inputs without changing targets or grippers."""
import math
from numbers import Real

import torch


class JointValueDropout:
    def __init__(self, probability=0.2):
        if (isinstance(probability, bool) or not isinstance(probability, Real)
                or not math.isfinite(probability) or not 0 <= probability <= 1):
            raise ValueError('Joint dropout probability must be a finite number in [0,1]')
        self.probability = float(probability)

    def __call__(self, state, eligible, *, seed, training=True):
        if (not isinstance(state, torch.Tensor) or state.device.type != 'cpu'
                or not state.is_floating_point() or state.ndim < 1
                or not torch.isfinite(state).all()):
            raise ValueError('Joint dropout expects finite normalized floating-point CPU state')
        if (not isinstance(eligible, torch.Tensor) or eligible.device.type != 'cpu'
                or eligible.dtype != torch.bool or eligible.shape != state.shape[-1:]):
            raise ValueError('Supply one boolean eligibility mask for the canonical state dimensions')
        dropped = torch.zeros_like(eligible)
        if not training or self.probability == 0 or not eligible.any():
            return state, dropped
        indices = eligible.nonzero(as_tuple=True)[0]
        generator = torch.Generator(device='cpu').manual_seed(seed)
        dropped[indices] = torch.rand(len(indices), generator=generator) < self.probability
        if not dropped.any():
            return state, dropped
        # Share one mask across a sequence; surviving values keep their scale.
        return state.masked_fill(dropped, 0), dropped
