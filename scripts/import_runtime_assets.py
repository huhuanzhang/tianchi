"""Copy byte-exact split/statistics from an existing project into ignored paths."""
import argparse, hashlib, json, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--from-root',required=True,type=Path)
    a=p.parse_args();names=['augmentation/artifacts/split.json','augmentation/artifacts/train_stats.json','training/artifacts/clean_train_norm.json']
    origin=a.from_root.expanduser().resolve();split=origin/names[0]
    split_hash=hashlib.sha256(split.read_bytes()).hexdigest()
    for name in names[1:]:
        if json.loads((origin/name).read_text())['split_sha256']!=split_hash:raise ValueError('Statistics do not match split')
    for name in names:
        src=origin/name;dst=ROOT/name
        if dst.exists() and dst.read_bytes()!=src.read_bytes():raise FileExistsError('Different runtime asset: '+str(dst))
    for name in names:
        src=origin/name;dst=ROOT/name;dst.parent.mkdir(parents=True,exist_ok=True)
        if src!=dst:shutil.copyfile(src,dst)
        print(dst)
if __name__=='__main__':main()
