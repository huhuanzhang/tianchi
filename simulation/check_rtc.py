"""Analytical checks for reverse-time RTC's sign, VJP, and padding mask."""
import torch
from types import SimpleNamespace
from .rtc_moe import batched_experts

from .rtc_guidance import get_prefix_weights, guided_velocity, openpi_guidance_weight


def main():
    weights = get_prefix_weights(2, 18, 50)
    assert torch.equal(weights[:2], torch.ones(2))
    assert torch.all(weights[2:18] > 0) and torch.all(weights[18:] == 0)
    assert torch.all(weights[:-1] >= weights[1:])
    assert openpi_guidance_weight(torch.tensor(1.), 5.).item() == 5.
    x = torch.zeros(1, 50, 4)
    target = torch.ones_like(x)
    mask = torch.tensor([1, 1, 0, 0], dtype=torch.bool)
    # v=0: the correct reverse-time correction decreases velocity; dt<0
    # therefore moves actions toward the positive old prefix.
    v = guided_velocity(lambda z: z*0, x, torch.tensor(.5), target, weights, 5., mask)
    next_x = x - .1*v
    assert torch.all(next_x[:, :2, :2] > 0)
    assert torch.all(next_x[:, :, 2:] == 0) and torch.all(next_x[:, 18:] == 0)
    # Coupled affine denoiser tests the full transposed Jacobian, including
    # cross-dimension coupling. Detaching v or using direct EMA fails this.
    matrix = torch.tensor([[.1, .2, .3, 0], [0, .2, .1, .1],
                           [.2, 0, .1, 0], [0, .1, 0, .2]])
    t = torch.tensor(.3)
    x = torch.randn(1, 50, 4, generator=torch.Generator().manual_seed(42))
    vel = x @ matrix.T
    denoised = x - t*vel
    error = (target-denoised)*weights[None, :, None]*mask[None, None, :]
    correction = error @ (torch.eye(4)-t*matrix)
    expected = vel-openpi_guidance_weight(t, 5.)*correction
    actual = guided_velocity(lambda z: z @ matrix.T, x, t, target, weights, 5., mask)
    torch.testing.assert_close(actual, expected)
    # Check the common differentiable MoE backend against an independent
    # per-token/per-expert implementation, both values and input gradients.
    gen = torch.Generator().manual_seed(17)
    expert = SimpleNamespace(gate_proj=torch.randn(3, 5, 4, generator=gen),
                             up_proj=torch.randn(3, 5, 4, generator=gen),
                             down_proj=torch.randn(3, 4, 5, generator=gen))
    z = torch.randn(2, 4, generator=gen).requires_grad_()
    router = torch.tensor([[.2, .8], [.6, .4]], requires_grad=True)
    chosen = torch.tensor([[2, 0], [1, 2]])
    out = batched_experts(expert, None, 3, router, chosen, z)
    reference = []
    for n in range(2):
        vals = []
        for k in range(2):
            e = int(chosen[n, k])
            g = torch.nn.functional.linear(z[n], expert.gate_proj[e])
            u = torch.nn.functional.linear(z[n], expert.up_proj[e])
            y = torch.nn.functional.linear(torch.nn.functional.silu(g)*u, expert.down_proj[e])
            vals.append(router[n, k]*y)
        reference.append(sum(vals))
    reference = torch.stack(reference)
    torch.testing.assert_close(out, reference)
    grads = torch.autograd.grad(out.sum(), (z, router))
    expected_grads = torch.autograd.grad(reference.sum(), (z, router))
    for a, b in zip(grads, expected_grads):
        torch.testing.assert_close(a, b)
    print("RTC checks passed: reverse-time sign, soft mask, padding, full VJP, MoE outputs and gradients")


if __name__ == "__main__":
    main()
