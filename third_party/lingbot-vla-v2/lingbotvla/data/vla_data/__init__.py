"""Load dataset implementations only when requested.

FeatureTransform can be used with the native clean reader without importing
the unrelated LeRobot dataset/decoder stack.
"""

__all__ = ["VLADataset", "MultiVLADataset"]


def __getattr__(name):
    if name == "VLADataset":
        from .base_dataset import VLADataset
        return VLADataset
    if name == "MultiVLADataset":
        from .multi_vla_dataset import MultiVLADataset
        return MultiVLADataset
    raise AttributeError(name)
