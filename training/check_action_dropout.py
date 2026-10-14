"""Check action-only dropout, inference parity, checkpointing, and fresh-base shapes."""
import argparse
import json
from pathlib import Path
import tempfile

import torch
from safetensors import safe_open
from torch.utils.checkpoint import checkpoint

from .action_dropout import normalize_config
from .native_model import ROOT, empty_model, make_config
from .rtc import assert_resume_compatible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='training/configs/default.json')
    parser.add_argument('--output', default='training/artifacts/action_dropout_check.json')
    args = parser.parse_args()
    settings = json.loads((ROOT / args.config).read_text())
    torch.set_num_threads(1)
    torch.manual_seed(532514)
    cfg = normalize_config(settings['action_expert_dropout'])
    for value in ({'ffn_output_probability': -1}, {'attention_output_probability': float('nan')},
                  {'attention_output_probability': 1}, {'typo': .1}, {'state_feature_probability': True}):
        try:
            normalize_config(value)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid dropout configuration accepted')
    different_probability = 0.0 if cfg['ffn_output_probability'] > 0 else .2
    changed = dict(settings, action_expert_dropout=dict(cfg, ffn_output_probability=different_probability))
    try:
        assert_resume_compatible(settings, changed)
    except ValueError:
        pass
    else:
        raise AssertionError('Resume accepted changed regularization')

    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import Qwen2DecoderLayer
    small = Qwen2Config(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                        num_attention_heads=2, num_key_value_heads=2, head_dim=4)
    small._attn_implementation = 'eager'
    small.action_attention_output_dropout = cfg['attention_output_probability']
    small.action_ffn_output_dropout = cfg['ffn_output_probability']
    layer = Qwen2DecoderLayer(small, 0)
    inputs, attended = torch.randn(2, 12, 8), torch.randn(2, 12, 8)
    def forward(x):
        return layer(x, attended, 0, x.shape[1], output_atten=True)[0]
    layer.eval()
    expected = forward(inputs)
    saved = (layer.attention_output_dropout.p, layer.ffn_output_dropout.p)
    layer.attention_output_dropout.p = layer.ffn_output_dropout.p = 0.
    torch.testing.assert_close(expected, forward(inputs), rtol=0, atol=0)
    layer.attention_output_dropout.p, layer.ffn_output_dropout.p = saved
    layer.train()
    first, second = forward(inputs), forward(inputs)
    assert not torch.equal(first, second)
    plain_input = inputs.clone().requires_grad_()
    torch.manual_seed(532514)
    plain = forward(plain_input)
    plain.square().mean().backward()
    gradients = {n: p.grad.clone() for n, p in layer.named_parameters() if p.grad is not None}
    layer.zero_grad(set_to_none=True)
    checked_input = inputs.clone().requires_grad_()
    torch.manual_seed(532514)
    checked = checkpoint(forward, checked_input, use_reentrant=False, preserve_rng_state=True)
    checked.square().mean().backward()
    torch.testing.assert_close(plain, checked, rtol=0, atol=0)
    torch.testing.assert_close(plain_input.grad, checked_input.grad, rtol=0, atol=0)
    for name, parameter in layer.named_parameters():
        if name in gradients:
            torch.testing.assert_close(gradients[name], parameter.grad, rtol=0, atol=0)

    model = empty_model(settings)
    assert model.config.action_expert_dropout == cfg
    layers = model.model.qwenvl_with_expert.qwen_expert.model.layers
    assert len(layers) == 36
    for expert in layers:
        assert expert.attention_output_dropout.p == cfg['attention_output_probability']
        assert expert.ffn_output_dropout.p == cfg['ffn_output_probability']
    frozen = model.model.qwenvl_with_expert.qwenvl
    assert all(not isinstance(module, torch.nn.Dropout) or module.p == 0 for module in frozen.modules())
    base = ROOT / settings['base_model']
    index = json.loads((base / 'model.safetensors.index.json').read_text())['weight_map']
    expected = model.state_dict()
    assert set(expected) == set(index), 'Dropout changed checkpoint tensor keys'
    count = 0
    for shard in sorted(set(index.values())):
        with safe_open(base / shard, framework='pt', device='cpu') as source:
            for name in source.keys():
                assert tuple(source.get_slice(name).get_shape()) == tuple(expected[name].shape), name
                count += 1
    assert count == len(expected)
    with tempfile.TemporaryDirectory() as directory:
        model.config.save_pretrained(directory)
        reload_settings = dict(settings, base_model=directory)
        reload_settings.pop('action_expert_dropout')
        assert make_config(reload_settings).action_expert_dropout == cfg
    report = dict(status='passed', config=args.config, action_expert_dropout=cfg,
                  expert_layers=len(layers), public_base_tensors_checked=count,
                  inference_dropout_disabled_and_matches_zero_probability=True,
                  training_dropout_active=True, checkpoint_recompute_outputs_and_gradients_match=True,
                  frozen_vlm_dropout_unchanged=True, config_export_reload_preserves_dropout=True,
                  resume_rejects_changed_dropout=True, model_weights_loaded=False,
                  four_gpu_training_tested=False)
    destination = ROOT / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
