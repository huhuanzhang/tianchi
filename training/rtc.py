"""Training-time action-prefix conditioning for Lingbot's reverse-time flow.

Reference: https://arxiv.org/abs/2512.05964. Here t=1 is noise and t=0
is clean, so known prefix tokens receive t=0 in both training and sampling.
Only released clean demonstrations provide the training prefix and targets.
"""
import json

import torch

METHOD = "action_prefix_v1"


def normalize_config(value, horizon=50):
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("rtc must be a configuration object")
    allowed = {"enabled", "method", "max_delay", "zero_delay_probability", "execute_steps"}
    if set(value) - allowed:
        raise ValueError("Unknown RTC settings: " + str(sorted(set(value) - allowed)))
    cfg = dict(enabled=False, method=METHOD, max_delay=12,
               zero_delay_probability=0.3, execute_steps=25)
    cfg.update(value)
    if not isinstance(cfg['enabled'], bool) or cfg['method'] != METHOD:
        raise ValueError("Unsupported RTC method or enable flag")
    if type(cfg['max_delay']) is not int or not 0 <= cfg['max_delay'] < horizon:
        raise ValueError("RTC max_delay must be an integer smaller than the prediction horizon")
    if not 0 <= cfg['zero_delay_probability'] <= 1:
        raise ValueError("RTC zero_delay_probability must be between zero and one")
    if type(cfg['execute_steps']) is not int or not 1 <= cfg['execute_steps'] <= horizon:
        raise ValueError("RTC execution interval must fit the prediction horizon")
    if cfg['enabled'] and cfg['execute_steps'] + cfg['max_delay'] >= horizon:
        raise ValueError("Leave queued actions beyond the execution interval and supported delay")
    return cfg


def assert_resume_compatible(saved_settings, current_settings):
    old = normalize_config(saved_settings.get('rtc'), saved_settings['chunk_size'])
    new = normalize_config(current_settings.get('rtc'), current_settings['chunk_size'])
    if old != new:
        raise ValueError("Resume RTC settings differ; initialize a new run from full weights to change the objective")
    from .action_dropout import normalize_config as normalize_action_dropout
    if normalize_action_dropout(saved_settings.get('action_expert_dropout')) != normalize_action_dropout(current_settings.get('action_expert_dropout')):
        raise ValueError('Resume action expert dropout differs; initialize a new run to change regularization')


def training_inputs(flow, actions, joint_mask, action_is_pad=None, *,
                    noise=None, time=None, delay=None):
    """Return native noise, token times, postfix supervision mask and delays."""
    batch, horizon, dimensions = actions.shape
    cfg = normalize_config(flow.config.rtc_training, horizon)
    if joint_mask is None or joint_mask.shape != actions.shape:
        raise ValueError("RTC requires the native per-action/per-joint supervision mask")
    valid = joint_mask.bool().clone()
    if action_is_pad is not None:
        if action_is_pad.shape != (batch, horizon):
            raise ValueError("RTC action padding mask has the wrong shape")
        valid &= ~action_is_pad.bool().unsqueeze(-1)
    # Dataset padding must be trailing. Leave at least one supervised action.
    valid_frames = valid.any(-1)
    lengths = valid_frames.sum(-1)
    expected = torch.arange(horizon, device=actions.device)[None] < lengths[:, None]
    if not torch.equal(valid_frames, expected) or (lengths == 0).any():
        raise ValueError("RTC needs a nonempty contiguous valid action prefix before trailing padding")
    if noise is None:
        noise = torch.randn_like(actions)
    if noise.shape != actions.shape:
        raise ValueError("RTC noise shape differs from the demonstration chunk")
    if time is None:
        time = flow.sample_time(batch, actions.device).to(actions.dtype)
    if time.shape != (batch,) or not torch.isfinite(time).all() or ((time < 0) | (time > 1)).any():
        raise ValueError("Supply one finite base flow time in [0,1] per sample")
    if delay is None:
        if cfg['max_delay'] == 0:
            delay = torch.zeros(batch, device=actions.device, dtype=torch.long)
        else:
            delay = torch.randint(1, cfg['max_delay'] + 1, (batch,), device=actions.device)
            unconditional = torch.rand(batch, device=actions.device) < cfg['zero_delay_probability']
            delay = torch.where(unconditional, 0, delay)
    else:
        delay = torch.as_tensor(delay, device=actions.device)
        if delay.ndim == 0:
            delay = delay.expand(batch)
        if delay.shape != (batch,) or delay.is_floating_point() or delay.dtype == torch.bool:
            raise ValueError("RTC delay must contain one integer per sample")
        if ((delay < 0) | (delay > cfg['max_delay'])).any():
            raise ValueError("Training delay is outside the configured range")
    delay = torch.minimum(delay.long(), lengths - 1)
    prefix = torch.arange(horizon, device=actions.device)[None] < delay[:, None]
    postfix = valid & ~prefix[:, :, None]
    # Exact native scalar-time path for a wholly unconditional batch.
    token_time = torch.where(prefix, 0.0, time[:, None]) if prefix.any() else time
    flow._rtc_last_delays = delay.detach()
    flow._rtc_last_supervised_counts = postfix.sum(dim=(1, 2)).detach()
    return noise, token_time, postfix, delay


@torch.no_grad()
def sample_prefix(flow, inputs, action_prefix, delay, action_mask, noise=None):
    """Clamp the committed prefix during each Euler step; no VJP guidance."""
    cfg = normalize_config(getattr(flow.config, 'rtc_training', None), flow.config.n_action_steps)
    if not cfg['enabled']:
        raise ValueError("Prefix sampling requires an RTC-trained checkpoint")
    state = inputs['state']
    batch, horizon, dimensions = state.shape[0], flow.config.n_action_steps, flow.config.max_action_dim
    delays = torch.as_tensor(delay, device=state.device)
    if delays.ndim == 0:
        delays = delays.expand(batch)
    if delays.shape != (batch,) or delays.is_floating_point() or delays.dtype == torch.bool:
        raise ValueError("Sampling delay must contain one integer per sample")
    if ((delays < 0) | (delays >= horizon)).any():
        raise ValueError("Sampling delay must be between zero and horizon-1")
    if not delays.any():
        return flow.sample_actions(**inputs, noise=None if noise is None else noise.clone())
    previous = torch.as_tensor(action_prefix, device=state.device, dtype=state.dtype)
    if previous.ndim != 3 or previous.shape[0] != batch or previous.shape[2] != dimensions:
        raise ValueError("Prefix must have shape [batch, available_actions, canonical_action_dim]")
    if previous.shape[1] > horizon or (delays > previous.shape[1]).any():
        raise ValueError("Known prefix is shorter than the committed delay")
    if not torch.isfinite(previous).all():
        raise ValueError("Prefix actions must be finite")
    real = torch.as_tensor(action_mask, device=state.device, dtype=torch.bool)
    if real.ndim == 1:
        real = real.expand(batch, -1)
    if real.shape != (batch, dimensions) or not real.any(-1).all():
        raise ValueError("Sampling requires the native real-joint mask")
    known = state.new_zeros((batch, horizon, dimensions))
    known[:, :previous.shape[1]] = previous * real[:, None]
    prefix = torch.arange(horizon, device=state.device)[None] < delays[:, None]
    if noise is None:
        noise = torch.randn_like(known)
    if noise.shape != known.shape or not torch.isfinite(noise).all():
        raise ValueError("Sampling noise has the wrong shape or nonfinite values")
    from lingbotvla.models.vla.lingbot_vla.utils import make_att_2d_masks
    (prefix_embs, prefix_pad_masks, prefix_att_masks, prefix_position_ids,
     visual_pos_masks, deepstack_visual_embeds) = flow.embed_prefix(
        inputs['images'], inputs['img_masks'], inputs['lang_tokens'], inputs['lang_masks'],
        image_grid_thw=inputs.get('image_grid_thw'))
    _, past_key_values, _ = flow.qwenvl_with_expert.forward(
        attention_mask=make_att_2d_masks(prefix_pad_masks, prefix_att_masks),
        position_ids=prefix_position_ids, vlm_position_ids=prefix_position_ids,
        past_key_values=None, inputs_embeds=[prefix_embs, None], use_cache=True,
        fill_kv_cache=True, visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds)
    x = noise.clone()
    dt = state.new_tensor(-1.0 / flow.config.num_steps)
    time = state.new_tensor(1.0)
    for _ in range(flow.config.num_steps):
        x = torch.where(prefix[:, :, None], known, x)
        token_time = torch.where(prefix, 0.0, time.expand(batch, horizon))
        velocity = flow.predict_velocity(state, prefix_pad_masks, past_key_values, x, token_time,
                                         prefix_position_ids=prefix_position_ids)
        x.add_(dt * velocity)
        time.add_(dt)
    # Euler would otherwise overwrite the prefix at the final step.
    return torch.where(prefix[:, :, None], known, x)


def export_rtc_assets(directory, config):
    cfg = normalize_config(getattr(config, 'rtc_training', None), config.n_action_steps)
    if cfg['enabled']:
        (directory/'rtc_config.json').write_text(json.dumps(dict(
            **cfg, prediction_horizon=config.n_action_steps, clean_flow_time=0,
            sampling='hard_prefix_each_euler_step', inference_vjp_guidance=False,
            delay_units='action commands consumed during inference'), indent=2) + '\n')
