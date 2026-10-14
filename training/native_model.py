"""Load every pretrained tensor into the native LingBot architecture."""
from pathlib import Path
import json
import sys

import torch
import yaml
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party/lingbot-vla-v2"
sys.path.insert(0, str(REPO))


def is_trainable(name):
    return name.startswith("model.qwenvl_with_expert.qwen_expert.") or name.startswith(
        ("model.state_proj.", "model.action_in_proj.", "model.action_out_proj.",
         "model.action_time_mlp_in.", "model.action_time_mlp_out.")
    )


def make_config(settings):
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
    template = yaml.safe_load((REPO / "configs/vla/robotwin/robotwin.yaml").read_text())
    values = template["train"]
    values.update(template["model"])
    # PyYAML treats bare 1e-3 as a string; the official typed CLI casts these.
    for name in ("sequence_wise_loss_coeff", "router_z_loss_coeff", "bias_update_speed", "routed_scaling_factor"):
        if name in values:
            values[name] = float(values[name])
    values.update(tokenizer_path=str(ROOT / settings["vlm_config"]),
                  vlm_repo_id=str(ROOT / settings["vlm_config"]),
                  train_expert_only=True, train_state_proj=True, freeze_vision_encoder=True,
                  attention_implementation="eager", vit_attn_implementation="sdpa",
                  moe_implementation="fused", action_fp32=settings["action_dtype"] == "float32",
                  chunk_size=settings["chunk_size"], use_compile=False,
                  precompute_grid_thw=False, use_qwen3_fixed_grid_cache=False, bc_only=True)
    config = LingbotVLAV2Config(**values)
    from .rtc import normalize_config
    base_config = ROOT/settings['base_model']/'config.json'
    saved = json.loads(base_config.read_text()) if base_config.is_file() else {}
    rtc = settings['rtc'] if 'rtc' in settings else saved.get('rtc_training')
    config.rtc_training = normalize_config(rtc, settings['chunk_size'])
    from .action_dropout import normalize_config as normalize_action_dropout
    config.action_expert_dropout = normalize_action_dropout(
        settings.get('action_expert_dropout', saved.get('action_expert_dropout')))
    config.use_cache = False
    return config


def empty_model(settings):
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch
    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import apply_lingbot_qwen2_patch
    from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import LingbotVlaV2Policy
    apply_lingbot_qwen3_vl_patch()
    apply_lingbot_qwen2_patch()
    config = make_config(settings)
    with init_empty_weights(include_buffers=False):
        model = LingbotVlaV2Policy(config)
    return model


def load_model(settings):
    model = empty_model(settings)
    base = ROOT / settings["base_model"]
    if model.config.rtc_training['enabled'] and (base/'clean_train_norm.json').is_file():
        from augmentation.prepare import file_hash
        if file_hash(base/'clean_train_norm.json') != file_hash(ROOT/settings['norm_stats']):
            raise ValueError('RTC continuation must preserve the source checkpoint normalization')
    expected = model.state_dict()
    upstream = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    missing = sorted(set(expected) - set(upstream))
    unexpected = sorted(set(upstream) - set(expected))
    if missing or unexpected:
        raise ValueError(f"Pretrained architecture mismatch: missing={missing[:12]}, unexpected={unexpected[:12]}")
    del expected
    action_dtype = getattr(torch, settings["action_dtype"])
    loaded = 0
    for shard in sorted(set(upstream.values())):
        with safe_open(base / shard, framework="pt", device="cpu") as source:
            for name in source.keys():
                value = source.get_tensor(name)
                dtype = action_dtype if is_trainable(name) else torch.bfloat16
                set_module_tensor_to_device(model, name, "cuda", value=value, dtype=dtype)
                loaded += 1
                del value
        print(f"Loaded official shard {shard} ({loaded} tensors total)", flush=True)
    remaining = [name for name, value in model.state_dict().items() if value.device.type == "meta"]
    if remaining:
        raise ValueError(f"Uninitialized tensors: {remaining[:12]}")
    model.to("cuda")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(is_trainable(name))
    if settings["checkpoint_gradient"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print("Pretrained tensors loaded with no missing or unexpected keys", flush=True)
    return model
