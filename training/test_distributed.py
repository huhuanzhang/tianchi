"""Check rank partitioning, resume cursors and real two-process DDP accumulation."""
from contextlib import nullcontext
import math
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from .distributed_utils import RankEpochSampler, epoch_indices, accumulation_steps, initialize_native_parallel


def ddp_worker(rank, path):
    dist.init_process_group("gloo",init_method=f"file://{path}",rank=rank,world_size=2)
    initialize_native_parallel(2, device_type="cpu")
    from lingbotvla.distributed.parallel_state import get_parallel_state
    state = get_parallel_state()
    assert state.world_size == 2 and state.dp_size == 2 and not state.ep_enabled
    layer=torch.nn.Linear(1,1,bias=False)
    with torch.no_grad(): layer.weight.fill_(0.5)
    model=DDP(layer)
    optimizer=torch.optim.SGD(model.parameters(),lr=0.1)
    optimizer.zero_grad()
    for micro in range(2):
        context=model.no_sync() if micro==0 else nullcontext()
        with context:
            value=torch.tensor([[float(micro*2+rank+1)]])
            loss=(model(value)-2*value).square().mean()/2
            loss.backward()
    optimizer.step()
    # Mean squared loss over x=1,2,3,4 has dL/dw=-22.5 at w=0.5.
    assert torch.allclose(layer.weight,torch.tensor([[2.75]]),atol=1e-6),layer.weight
    dist.destroy_process_group()


class DistributedTests(unittest.TestCase):
    def test_all_one_to_eight_gpu_partitions_and_resume(self):
        for world in range(1,9):
            batch=math.ceil(64/world)*world
            accum=accumulation_steps(batch,1,world)
            expected=epoch_indices(257,532514,2,batch)
            ranks=[list(RankEpochSampler(257,532514,2,batch,1,r,world)) for r in range(world)]
            recovered=[]
            for update in range(len(expected)//batch):
                for micro in range(accum):
                    for rank in range(world):
                        recovered.append(ranks[rank][update*accum+micro])
            self.assertEqual(recovered,expected)
            self.assertEqual(set(expected),set(range(257)))
            for rank in range(world):
                resumed=list(RankEpochSampler(257,532514,2,batch,1,rank,world,skip_updates=2))
                self.assertEqual(resumed,ranks[rank][2*accum:])

    def test_epoch_changes_and_batched_partition(self):
        self.assertNotEqual(epoch_indices(101,9,0,64),epoch_indices(101,9,1,64))
        expected=epoch_indices(101,9,0,64)
        ranks=[list(RankEpochSampler(101,9,0,64,2,r,4)) for r in range(4)]
        recovered=[]
        for update in range(2):
            for micro in range(8):
                for rank in range(4):
                    position=update*16+micro*2
                    recovered.extend(ranks[rank][position:position+2])
        self.assertEqual(recovered,expected)

    def test_two_process_gradient_accumulation(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(ddp_worker,args=(f"{directory}/rendezvous",),nprocs=2,join=True)


if __name__ == "__main__":
    unittest.main()
