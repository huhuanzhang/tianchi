"""A common differentiable FP32 MoE backend for RTC and its controls.

Upstream's no-grad Triton path has no VJP, while its autograd fused path casts
to BF16. Switching between them would confound RTC with a precision change.
Use the same vectorized PyTorch expert computation in every experiment arm.
Only runtime functions change; all model parameters stay identical and frozen.
"""
from types import MethodType

import torch
from torch.nn import functional as F


def batched_experts(self, module, num_experts, routing_weights, selected_experts, hidden_states):
    x = hidden_states.unsqueeze(0).expand(num_experts, -1, -1)
    gate = torch.bmm(x, self.gate_proj.transpose(1, 2))
    up = torch.bmm(x, self.up_proj.transpose(1, 2))
    values = torch.bmm(F.silu(gate)*up, self.down_proj.transpose(1, 2))
    # Select the existing router's chosen experts; no new routing rule.
    values = values.permute(1, 0, 2)
    selected = values.gather(1, selected_experts[:, :, None].expand(-1, -1, values.shape[-1]))
    return (selected*routing_weights[:, :, None]).sum(dim=1)


def install(model):
    import lingbotvla.models.vla.lingbot_vla.qwen2_action_expert as expert_module
    expert_module.robby_moe_forward = None
    count = 0
    for module in model.modules():
        if isinstance(module, expert_module.Qwen2FusedExperts):
            module.forward = MethodType(batched_experts, module)
            count += 1
    if count != 36:
        raise ValueError(f"Unexpected Lingbot expert layer count: {count}")
    return count
