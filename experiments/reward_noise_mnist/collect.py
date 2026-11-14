"""Bundle every run's log into one small file the article's plot script reads.

    python collect.py      ->  results_summary.json  (+ copies final sample grids to grids/)
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
out, grids = {}, ROOT / 'grids'
grids.mkdir(exist_ok=True)
for run in sorted((ROOT / 'runs').glob('*')):
    log = run / 'log.jsonl'
    if not log.exists():
        continue
    rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    out[run.name] = {'config': json.loads((run / 'config.json').read_text()),
                     'finished': (run / 'final.pt').exists(), 'log': rows}
    finals = sorted(run.glob('samples_*.png'))
    if finals:
        shutil.copy(finals[-1], grids / f'{run.name}.png')
cls = ROOT / 'ckpt' / 'classifiers.json'
summary = {'runs': out, 'classifiers': json.loads(cls.read_text()) if cls.exists() else None}
(ROOT / 'results_summary.json').write_text(json.dumps(summary))
print(f'{len(out)} runs -> results_summary.json')
