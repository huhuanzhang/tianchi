"""Create an episode holdout and compute statistics from clean train frames."""
from pathlib import Path
import argparse
import hashlib
import json
import random

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/RoboTwin_lerobot_v21")
    parser.add_argument("--output", type=Path, default=ROOT / "augmentation/artifacts")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=532514)
    args = parser.parse_args()
    if not 0 <= args.validation_fraction < 1:
        parser.error("validation-fraction must be in [0,1)")
    dataset = args.dataset.resolve()
    info = json.loads((dataset / "meta/info.json").read_text())
    episodes = [json.loads(s) for s in (dataset / "meta/episodes.jsonl").read_text().splitlines() if s]
    if len(episodes) != 2500 or info["total_episodes"] != 2500:
        raise ValueError("Expected the official 2500-episode clean export")
    episode_by_id = {e["episode_index"]: e for e in episodes}
    if set(episode_by_id) != set(range(2500)):
        raise ValueError("Unexpected episode IDs")
    ids = sorted(episode_by_id)
    random.Random(args.seed).shuffle(ids)
    count = round(len(ids) * args.validation_fraction)
    if args.validation_fraction > 0:
        count = max(1, count)
    val_ids, train_ids = sorted(ids[:count]), sorted(ids[count:])
    keys = ("observation.state", "action")
    sums = {k: np.zeros(14, np.float64) for k in keys}
    sums_sq = {k: np.zeros(14, np.float64) for k in keys}
    minima = {k: np.full(14, np.inf) for k in keys}
    maxima = {k: np.full(14, -np.inf) for k in keys}
    train_frames = 0
    for number, idx in enumerate(train_ids, 1):
        parquet = dataset / info["data_path"].format(episode_index=idx, episode_chunk=idx // info["chunks_size"])
        table = pq.read_table(parquet, columns=list(keys) + ["episode_index"])
        if table.num_rows != episode_by_id[idx]["length"] or set(table["episode_index"].to_pylist()) != {idx}:
            raise ValueError(f"Episode metadata mismatch in {parquet}")
        for key in keys:
            values = np.asarray(table[key].to_pylist(), dtype=np.float64)
            if values.shape != (table.num_rows, 14) or not np.isfinite(values).all():
                raise ValueError(f"Invalid {key} in {parquet}")
            sums[key] += values.sum(axis=0)
            sums_sq[key] += np.square(values).sum(axis=0)
            minima[key] = np.minimum(minima[key], values.min(axis=0))
            maxima[key] = np.maximum(maxima[key], values.max(axis=0))
        train_frames += table.num_rows
        if number % 500 == 0:
            print(f"Read only clean train episodes: {number}/{len(train_ids)}", flush=True)
    features = {}
    for key in keys:
        mean = sums[key] / train_frames
        variance = np.maximum(sums_sq[key] / train_frames - mean**2, 0)
        features[key] = {"mean": mean.tolist(), "std": np.sqrt(variance).tolist(),
                         "min": minima[key].tolist(), "max": maxima[key].tolist()}
    split = {"dataset_path": str(dataset), "seed": args.seed,
             "strategy": "global_random_whole_episode_holdout",
             "task_stratified": False,
             "note": "task_index indexes language strings; a verified 50-task episode mapping is not available.",
             "train_episodes": train_ids, "validation_episodes": val_ids,
             "train_frames": train_frames,
             "validation_frames": sum(episode_by_id[i]["length"] for i in val_ids)}
    args.output.mkdir(parents=True, exist_ok=True)
    split_path = args.output / "split.json"
    split_path.write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n")
    stats = {"dataset_path": str(dataset), "source": "official_clean_training_episodes_only",
             "split_sha256": file_hash(split_path), "frame_count": train_frames,
             "episode_count": len(train_ids), "std_ddof": 0,
             "raw_joint_indices": list(range(6)) + list(range(7, 13)),
             "raw_gripper_indices": [6, 13], "features": features}
    (args.output / "train_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({"train_episodes": len(train_ids), "validation_episodes": len(val_ids),
                      "train_frames": train_frames, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
