"""Write ignored runtime paths while preserving the versioned recipe."""
import argparse, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def main():
    versions=json.loads((ROOT/'versions.json').read_text())
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=list(versions), default=json.loads((ROOT/'VERSION.json').read_text())['profile'])
    p.add_argument('--base-model');p.add_argument('--dataset');p.add_argument('--output')
    p.add_argument('--destination', default='configs/runtime.json')
    a=p.parse_args();settings=json.loads((ROOT/versions[a.profile]['config']).read_text())
    for key,value in [('base_model',a.base_model),('dataset_root',a.dataset),('output',a.output)]:
        if value is not None:settings[key]=str(Path(value).expanduser().resolve())
    target=ROOT/a.destination;target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(settings,indent=2)+'\n');print(target)
if __name__=='__main__':main()
