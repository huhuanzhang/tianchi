"""Deterministic global batches shared by one to eight training ranks."""
import math

import torch
from torch.utils.data import Sampler


def initialize_native_parallel(world_size, device_type="cuda"):
    # The native fused MoE queries LingBot's parallel state independently of DDP.
    # Importing native_model also makes the fixed upstream source importable.
    from . import native_model
    from lingbotvla.distributed.parallel_state import init_parallel_state
    init_parallel_state(dp_size=world_size, dp_replicate_size=world_size,
                        dp_shard_size=1, dp_mode="ddp", device_type=device_type)


def accumulation_steps(global_batch, micro_batch, world_size):
    denominator = micro_batch * world_size
    if min(global_batch, micro_batch, world_size) < 1 or global_batch % denominator:
        raise ValueError("global_batch must be divisible by micro_batch * GPU count")
    return global_batch // denominator


def epoch_indices(length, seed, epoch, global_batch):
    if length < 1:
        raise ValueError("Training split is empty")
    indices = torch.randperm(length, generator=torch.Generator().manual_seed(seed + epoch)).tolist()
    padded = math.ceil(length / global_batch) * global_batch
    repeats = padded - length
    indices.extend((indices * math.ceil(repeats / length))[:repeats])
    return indices


class RankEpochSampler(Sampler):
    """Keep the same ordered global batches when GPU count changes.

    Epoch tails repeat at most global_batch-1 frames to maintain synchronized
    updates. Resume is expressed in global updates, independent of worker prefetch.
    """
    def __init__(self, length, seed, epoch, global_batch, micro_batch, rank, world_size, skip_updates=0):
        accumulation_steps(global_batch, micro_batch, world_size)
        if not 0 <= rank < world_size:
            raise ValueError("Invalid rank")
        self.indices = []
        global_indices = epoch_indices(length, seed, epoch, global_batch)
        for start in range(skip_updates * global_batch, len(global_indices), global_batch):
            for offset in range(0, global_batch, micro_batch * world_size):
                position = start + offset + rank * micro_batch
                self.indices.extend(global_indices[position:position + micro_batch])

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
