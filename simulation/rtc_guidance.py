"""Inference-only RTC, ported from the user's OpenPI rtc.py (reverse flow time).

Reference: rlt_openpi_franka_v4.5/third_party/openpi/src/openpi/models/rtc.py.
No model parameter gradients, optimizer, additional weights, or expert actions.
"""
import torch


def get_prefix_weights(inference_delay, prefix_attention_horizon, action_horizon,
                       schedule="exp", *, device=None):
    if not 0 <= inference_delay <= prefix_attention_horizon <= action_horizon:
        raise ValueError("RTC requires 0 <= delay <= prefix horizon <= chunk length")
    indices = torch.arange(action_horizon, device=device)
    start, end = inference_delay, prefix_attention_horizon
    if schedule == "ones":
        weights = torch.ones(action_horizon, device=device)
    elif schedule == "zeros":
        weights = (indices < start).float()
    elif schedule in ("linear", "exp"):
        weights = ((start - 1 - indices) / max(end-start+1, 1) + 1).clamp(0, 1)
        if schedule == "exp":
            weights = weights * torch.expm1(weights) / (torch.e - 1)
    else:
        raise ValueError(f"Unknown prefix schedule: {schedule}")
    return torch.where(indices >= end, 0, weights).float()


def openpi_guidance_weight(time, max_guidance_weight):
    tau = 1 - time
    inv_r2 = (tau.square() + time.square()) / time.square().clamp_min(1e-12)
    weight = time / tau.clamp_min(1e-12) * inv_r2
    return torch.nan_to_num(weight, nan=0, posinf=max_guidance_weight,
                            neginf=0).clamp(max=max_guidance_weight)


def guided_velocity(velocity_fn, x_t, time, previous_actions, prefix_weights,
                    max_guidance_weight, action_mask):
    # Re-enable input differentiation only for one Euler step. Never retain
    # graphs across denoising steps, and never populate parameter .grad.
    with torch.enable_grad():
        x = x_t.detach().requires_grad_(True)
        velocity = velocity_fn(x)
        denoised = x - time * velocity
        error = ((previous_actions - denoised) * prefix_weights[None, :, None]
                 * action_mask[None, None, :]).detach()
        correction = torch.autograd.grad(denoised, x, grad_outputs=error,
                                         create_graph=False)[0]
    strength = openpi_guidance_weight(time, max_guidance_weight)
    # Euler integration uses negative dt. This sign pulls the clean estimate
    # toward the old prefix, just as the user's reverse-time JAX version does.
    return (velocity - strength * correction).detach()


@torch.no_grad()
def sample_actions_rtc(flow, inputs, noise, previous_actions, action_mask, *,
                       inference_delay=2, prefix_attention_horizon=18,
                       max_guidance_weight=5.0, schedule="exp"):
    from lingbotvla.models.vla.lingbot_vla.utils import make_att_2d_masks

    if torch.is_inference_mode_enabled():
        raise RuntimeError("RTC VJP needs no_grad rather than inference_mode")
    if any(parameter.requires_grad for parameter in flow.parameters()):
        raise ValueError("RTC evaluation requires frozen model parameters")
    state = inputs["state"]
    (prefix_embs, prefix_pad_masks, prefix_att_masks, prefix_position_ids,
     visual_pos_masks, deepstack_visual_embeds) = flow.embed_prefix(
        inputs["images"], inputs["img_masks"], inputs["lang_tokens"],
        inputs["lang_masks"], image_grid_thw=inputs.get("image_grid_thw"))
    _, past_key_values, _ = flow.qwenvl_with_expert.forward(
        attention_mask=make_att_2d_masks(prefix_pad_masks, prefix_att_masks),
        position_ids=prefix_position_ids, vlm_position_ids=prefix_position_ids,
        past_key_values=None, inputs_embeds=[prefix_embs, None], use_cache=True,
        fill_kv_cache=True, visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds)
    weights = get_prefix_weights(inference_delay, prefix_attention_horizon,
                                 noise.shape[1], schedule, device=state.device)
    x_t = noise.clone()
    dt = state.new_tensor(-1.0 / flow.config.num_steps)
    time = state.new_tensor(1.0)
    while time >= -dt / 2:
        def velocity_fn(x):
            return flow.predict_velocity(
                state, prefix_pad_masks, past_key_values, x,
                time.expand(state.shape[0]), prefix_position_ids=prefix_position_ids)
        if previous_actions is None or max_guidance_weight == 0:
            velocity = velocity_fn(x_t)
        else:
            velocity = guided_velocity(velocity_fn, x_t, time, previous_actions,
                                       weights, max_guidance_weight, action_mask)
        # Match the native sampler's Euler update and scalar time storage.
        x_t.add_(dt * velocity)
        time.add_(dt)
    if any(parameter.grad is not None for parameter in flow.parameters()):
        raise RuntimeError("RTC unexpectedly accumulated a model parameter gradient")
    return x_t
