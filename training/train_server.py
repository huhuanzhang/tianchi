"""Native LingBot clean-only BC with online augmentation and single-node DDP."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from safetensors.torch import load_file

from .clean_dataset import CleanDataset
from .distributed_utils import RankEpochSampler, accumulation_steps, initialize_native_parallel
from .native_model import ROOT, load_model
from .train_local import export_runtime_assets, gpu_batch, save_checkpoint, validate
from augmentation.prepare import file_hash
from .rtc import normalize_config, assert_resume_compatible


class LossOnly(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, **batch):
        return self.policy(**batch)[0]


def worker_init(_):
    torch.set_num_threads(1)
    import pyarrow as pa
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)


def make_loader(dataset, settings, rank, world, epoch, skip_updates):
    dataset.epoch = epoch
    sampler = RankEpochSampler(len(dataset), settings["seed"], epoch, settings["global_batch_size"],
                               settings["micro_batch_size"], rank, world, skip_updates)
    workers = settings["num_workers"]
    options = {"multiprocessing_context": "spawn", "prefetch_factor": 2,
               "worker_init_fn": worker_init} if workers else {}
    return DataLoader(dataset, sampler=sampler, batch_size=settings["micro_batch_size"],
                      num_workers=workers, pin_memory=settings.get("pin_memory", True),
                      generator=torch.Generator().manual_seed(settings["seed"] + epoch + rank), **options)


def close_loader(iterator):
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if shutdown is None or getattr(iterator, "_shutdown", False):
        return
    # Stop submitting reads, then drain the bounded prefetch queue. This lets
    # shared-tensor producers finish before their multiprocessing queues close.
    iterator._sampler_iter = iter(())
    try:
        for _ in iterator:
            pass
    finally:
        shutdown()


def learning_rate(settings, step, total):
    warmup = min(settings["warmup_steps"], max(1, total // 10))
    if step <= warmup:
        return settings["learning_rate"] * step / warmup
    fraction = (step - warmup) / max(1, total - warmup)
    return settings["min_learning_rate"] + 0.5 * (settings["learning_rate"] - settings["min_learning_rate"]) * (1 + math.cos(math.pi * fraction))


def retain_checkpoints(output, count):
    if count < 1:
        return
    directories = sorted(path for path in output.glob("step_*") if (path / "metadata.json").is_file())
    for path in directories[:-count]:
        shutil.rmtree(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "training/configs/default.json")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--global-batch", type=int)
    parser.add_argument("--micro-batch", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument('--expert-dropout', type=float, help='Training-only dropout on both action-expert residual branches')
    parser.add_argument('--state-feature-dropout', type=float, help='Training-only whole encoded-state masking probability')
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-save", action="store_true", help="Bounded benchmark/check only")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    settings['rtc'] = normalize_config(settings.get('rtc'), settings['chunk_size'])
    if args.expert_dropout is not None or args.state_feature_dropout is not None:
        from .action_dropout import normalize_config as normalize_action_dropout
        dropout = normalize_action_dropout(settings.get('action_expert_dropout'))
        if args.expert_dropout is not None:
            dropout.update(attention_output_probability=args.expert_dropout,
                           ffn_output_probability=args.expert_dropout)
        if args.state_feature_dropout is not None:
            dropout['state_feature_probability'] = args.state_feature_dropout
        settings['action_expert_dropout'] = normalize_action_dropout(dropout)
    for key, value in (("num_train_epochs", args.epochs), ("max_steps", args.max_steps),
                       ("global_batch_size", args.global_batch), ("micro_batch_size", args.micro_batch),
                       ("num_workers", args.workers)):
        if value is not None:
            settings[key] = value
    rank, world = int(os.getenv("RANK", "0")), int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if not 1 <= world <= 8 or settings["num_workers"] < 0 or settings["num_train_epochs"] < 1:
        parser.error("Supported: 1-8 GPUs, nonnegative workers, positive epochs")
    requested_batch = settings["global_batch_size"]
    if requested_batch < 1 or settings["micro_batch_size"] < 1:
        parser.error("Batch sizes must be positive")
    denominator = settings["micro_batch_size"] * world
    settings["global_batch_size"] = math.ceil(requested_batch / denominator) * denominator
    accum = accumulation_steps(settings["global_batch_size"], settings["micro_batch_size"], world)
    output = args.output or ROOT / settings["output"]
    if not args.resume and output.exists() and (
            any((output / name).exists() for name in ('metrics.jsonl', 'run_config.json', 'report.json'))
            or any(output.glob('step_*'))):
        parser.error('Output already contains a run; use --resume or choose a new --output')
    if not torch.cuda.is_available():
        parser.error("CUDA GPU is required")
    torch.cuda.set_device(local_rank)
    if "RANK" in os.environ:
        dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    distributed = dist.is_initialized()
    if distributed:
        initialize_native_parallel(world)
    settings["rank"] = rank
    settings["gradient_accumulation_steps"] = accum
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(settings["torch_threads"])
    torch.manual_seed(settings["seed"] + rank)
    torch.cuda.manual_seed_all(settings["seed"] + rank)
    torch.cuda.reset_peak_memory_stats()
    policy = load_model(settings)
    settings['action_expert_dropout'] = policy.config.action_expert_dropout
    training = CleanDataset(settings, policy.config, training=True)
    holdout = CleanDataset(settings, policy.config, training=False)
    updates_per_epoch = math.ceil(len(training) / settings["global_batch_size"])
    total = settings.get("max_steps") or updates_per_epoch * settings["num_train_epochs"]
    if total < 1:
        parser.error("max-steps must be positive")
    parameters = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"], weight_decay=settings["weight_decay"], fused=True)
    start_step = 0
    if args.resume:
        metadata = json.loads((args.resume / "metadata.json").read_text())
        assert_resume_compatible(metadata['settings'], settings)
        for key, setting in (("split_sha256", "split"), ("augment_config_sha256", "augment_config"), ("native_norm_sha256", "norm_stats")):
            if metadata[key] != file_hash(ROOT / settings[setting]):
                raise ValueError(f"Resume {setting} differs from checkpoint")
        if metadata["settings"].get("global_batch_size") != settings["global_batch_size"]:
            raise ValueError("Keep global batch fixed for cursor and learning-rate resume")
        state = load_file(args.resume / "trainable.safetensors", device="cpu")
        expected = {name for name, parameter in policy.named_parameters() if parameter.requires_grad}
        if set(state) != expected:
            raise ValueError("Resume trainable tensor keys differ")
        with torch.no_grad():
            for name, parameter in policy.named_parameters():
                if name in state:
                    parameter.copy_(state[name].to(parameter.device))
        del state
        optimizer.load_state_dict(torch.load(args.resume / "optimizer.pt", map_location="cuda", weights_only=True))
        rng_path = args.resume / f"rng_rank_{rank:02d}.pt"
        if metadata.get("world_size") == world and rng_path.is_file():
            rng = torch.load(rng_path, weights_only=True)
            torch.set_rng_state(rng["cpu"])
            torch.cuda.set_rng_state_all(rng["cuda"])
        start_step = metadata["step"]
    if total <= start_step:
        parser.error("Target max-steps must exceed resumed step")
    wrapper = LossOnly(policy)
    model = DDP(wrapper, device_ids=[local_rank], broadcast_buffers=False,
                find_unused_parameters=True, gradient_as_bucket_view=True) if distributed else wrapper
    model.train()
    settings.update(output=str(output), world_size=world)
    if rank == 0:
        (output / "run_config.json").write_text(json.dumps(settings, indent=2) + "\n")
        print(json.dumps({"world_size": world, "requested_global_batch": requested_batch,
                          "global_batch": settings["global_batch_size"],
                          "micro_batch_per_gpu": settings["micro_batch_size"], "accumulation": accum,
                          "updates_per_epoch": updates_per_epoch, "target_updates": total,
                          "action_expert_dropout": settings['action_expert_dropout'],
                          "equivalent_epochs": total / updates_per_epoch}), flush=True)
    stop_requested = False
    def request_stop(*_):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    epoch = start_step // updates_per_epoch
    loader = make_loader(training, settings, rank, world, epoch, start_step % updates_per_epoch)
    iterator = iter(loader)
    timings = []
    last_step = start_step
    interrupted = False

    def checkpoint(step):
        if args.no_save:
            return
        directory = output / f"step_{step:06d}"
        if rank == 0:
            save_checkpoint(policy, optimizer, directory, settings, step, step // updates_per_epoch,
                            True, (step % updates_per_epoch) * settings["global_batch_size"])
            path = directory / "metadata.json"
            metadata = json.loads(path.read_text())
            metadata.update(world_size=world, updates_per_epoch=updates_per_epoch,
                            update_in_epoch=step % updates_per_epoch, total_target_updates=total,
                            global_samples_seen=step * settings["global_batch_size"])
            path.write_text(json.dumps(metadata, indent=2) + "\n")
        if distributed:
            dist.barrier()
        torch.save({"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
                   directory / f"rng_rank_{rank:02d}.pt")
        if distributed:
            dist.barrier()
        if rank == 0:
            retain_checkpoints(output, settings["keep_checkpoints"])

    try:
        for step in range(start_step + 1, total + 1):
            torch.cuda.synchronize()
            began = time.perf_counter()
            lr = learning_rate(settings, step, total)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            dropped_joints = eligible_joints = 0
            cropped_views = eligible_views = 0
            rtc_statistics = torch.zeros(4, device='cuda', dtype=torch.float64)
            for micro in range(accum):
                try:
                    sample = next(iterator)
                except StopIteration:
                    epoch += 1
                    loader = make_loader(training, settings, rank, world, epoch, 0)
                    iterator = iter(loader)
                    sample = next(iterator)
                dropped_joints += int(sample['joint_dropout_count'].sum())
                eligible_joints += int(sample['joint_dropout_eligible_count'].sum())
                cropped_views += int(sample['image_crop_count'].sum())
                eligible_views += int(sample['image_crop_eligible_count'].sum())
                context = model.no_sync() if distributed and micro < accum - 1 else nullcontext()
                with context:
                    loss = model(**gpu_batch(sample))
                    finite = torch.isfinite(loss).to(torch.int32)
                    if distributed:
                        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                    if not finite.item():
                        raise ValueError("Nonfinite BC loss on a rank")
                    loss_sum += loss.detach().item()
                    if settings['rtc']['enabled']:
                        delays = policy.model._rtc_last_delays
                        rtc_statistics[0] += delays.sum()
                        rtc_statistics[1] += delays.numel()
                        rtc_statistics[2] += (delays == 0).sum()
                        rtc_statistics[3] += policy.model._rtc_last_supervised_counts.sum()
                    (loss / accum).backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, foreach=False)
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite gradient norm")
            optimizer.step()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - began
            statistics = torch.tensor([loss_sum / accum, seconds, torch.cuda.max_memory_allocated() / 2**30,
                                       dropped_joints, eligible_joints, cropped_views, eligible_views],
                                      device="cuda", dtype=torch.float64)
            if distributed:
                loss_metric = statistics[:1].clone()
                dist.all_reduce(loss_metric)
                dist.all_reduce(statistics[1:3], op=dist.ReduceOp.MAX)
                dist.all_reduce(statistics[3:])
                statistics[0] = loss_metric[0] / world
                if settings['rtc']['enabled']:
                    dist.all_reduce(rtc_statistics)
            timings.append(statistics[1].item())
            last_step = step
            if rank == 0:
                record = {"step": step, "epoch_progress": step / updates_per_epoch,
                          "loss": statistics[0].item(), "learning_rate": lr,
                          "gradient_norm": norm.item(), "seconds": statistics[1].item(),
                          "peak_vram_gib": statistics[2].item(),
                          "joint_dropout_fraction": (statistics[3]/statistics[4].clamp(min=1)).item(),
                          "image_crop_view_fraction": (statistics[5]/statistics[6].clamp(min=1)).item(),
                          "remaining_training_hours": sum(timings[-50:]) / len(timings[-50:]) * (total - step) / 3600}
                if settings['rtc']['enabled']:
                    record.update(rtc_delay_mean=(rtc_statistics[0]/rtc_statistics[1]).item(),
                                  rtc_unconditional_fraction=(rtc_statistics[2]/rtc_statistics[1]).item(),
                                  rtc_supervised_entries=rtc_statistics[3].item())
                print(json.dumps(record), flush=True)
                with (output / "metrics.jsonl").open("a") as file:
                    file.write(json.dumps(record) + "\n")
            stopped = torch.tensor(int(stop_requested), device="cuda")
            if distributed:
                dist.all_reduce(stopped, op=dist.ReduceOp.MAX)
            interrupted = bool(stopped.item())
            optimizer.zero_grad(set_to_none=True)
            if interrupted or step == total or step % settings["save_steps"] == 0:
                checkpoint(step)
            if interrupted:
                break
            if step % settings["validation_steps"] == 0 and len(holdout):
                if rank == 0:
                    rng_cpu, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all()
                    loss = validate(policy, holdout, settings["validation_batches"], settings["seed"] + 1)
                    torch.set_rng_state(rng_cpu)
                    torch.cuda.set_rng_state_all(rng_cuda)
                    with (output / "validation.jsonl").open("a") as file:
                        file.write(json.dumps({"step": step, "bc_loss": loss,
                                               "rtc_validation_delay": 0 if settings['rtc']['enabled'] else None}) + "\n")
                if distributed:
                    dist.barrier()
        close_loader(iterator)
        if rank == 0:
            full_export = None
            if not args.no_save and settings["export_full"]:
                full = output / f"step_{last_step:06d}" / "hf_ckpt"
                policy.save_pretrained(full, max_shard_size="2GB", safe_serialization=True)
                export_runtime_assets(full, settings, policy.config)
                full_export = str(full)
            report = {"status": "interrupted_checkpoint_saved" if interrupted else "passed",
                      "completed_step": last_step, "world_size": world,
                      "target_updates": total, "updates_per_epoch": updates_per_epoch,
                      "effective_batch_size": settings["global_batch_size"],
                      "mean_update_seconds": sum(timings) / len(timings),
                      "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "full_export": full_export, "weights_saved": not args.no_save}
            (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        # Close prefetch workers and the pin-memory thread before CUDA/Python
        # teardown; relying on __del__ can abort workers during interpreter exit.
        close_loader(iterator)
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
