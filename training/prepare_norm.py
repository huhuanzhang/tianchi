"""Compute native LingBot normalization statistics from the clean train split."""
from pathlib import Path
import argparse
import json

import numpy as np
import pyarrow.parquet as pq

from augmentation.prepare import ROOT, file_hash
from augmentation.clean_augment import JOINTS, GRIPPERS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=ROOT / "augmentation/artifacts/split.json")
    parser.add_argument("--output", type=Path, default=ROOT / "training/artifacts/clean_train_norm.json")
    args = parser.parse_args()
    split_path = args.split
    split = json.loads(split_path.read_text())
    dataset = Path(split["dataset_path"])
    info = json.loads((dataset / "meta/info.json").read_text())
    columns = {"observation.state": [], "action": []}
    for idx in split["train_episodes"]:
        path = dataset / info["data_path"].format(episode_index=idx, episode_chunk=idx // info["chunks_size"])
        table = pq.read_table(path, columns=list(columns))
        for key in columns:
            columns[key].append(np.asarray(table[key].to_pylist(), dtype=np.float32))
    norms = {}
    for raw_key, arrays in columns.items():
        all_values = np.concatenate(arrays)
        if not np.isfinite(all_values).all():
            raise ValueError("Nonfinite clean values")
        for group, indices in (("arm.position", JOINTS), ("effector.position", GRIPPERS)):
            values = all_values[:, list(indices)]
            name = f"{raw_key}.{group}"
            norms[name] = {"mean": values.mean(0, dtype=np.float64).tolist(),
                           "std": values.std(0, dtype=np.float64).tolist(),
                           "min": values.min(0).tolist(), "max": values.max(0).tolist(),
                           "q01": np.quantile(values, 0.01, axis=0).tolist(),
                           "q99": np.quantile(values, 0.99, axis=0).tolist()}
    result = {"norm_stats": norms, "count": split["train_frames"],
              "source": "official_clean_training_episodes_only",
              "split_sha256": file_hash(split_path), "action_statistics": "single_frame_absolute_actions"}
    path = args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print("Wrote clean-only native normalization:", path, flush=True)


if __name__ == "__main__":
    main()
