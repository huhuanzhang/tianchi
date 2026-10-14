"""Training regularization for the action expert; no action-label dropout."""
import math


def normalize_config(value):
    defaults = dict(attention_output_probability=0.0, ffn_output_probability=0.0,
                    state_feature_probability=0.0)
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) - set(defaults):
        raise ValueError('Unknown action_expert_dropout settings')
    result = {**defaults, **value}
    for name, probability in result.items():
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise ValueError(f'{name} must be a numeric probability')
        if not math.isfinite(probability) or not 0 <= probability < 1:
            raise ValueError(f'{name} must be finite and in [0, 1)')
        result[name] = float(probability)
    return result
