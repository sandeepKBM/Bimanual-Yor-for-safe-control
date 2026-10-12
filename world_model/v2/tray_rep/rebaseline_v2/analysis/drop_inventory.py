"""Appends an inventory of every dropped F5/F5P carry (mu in 1e-3) in expA/expB (NEW) to RESULT_REBASELINE.md and summary_rebaseline.json. SAVED DATA ONLY."""
import glob, json, re
from pathlib import Path
from world_model.v2.tray_rep.bridge import run as R
RB = Path('/common/users/ss5772/bimanual_data/tray_rep/rebaseline_v2')
inv = {}
for p in sorted(glob.glob(str(RB / 'sets_expA/main/test/*/F5__*.jsonl')) + glob.glob(str(RB / 'vlmprior_expB/main/test/*/*.jsonl'))):
    d = [(r['ep'], round(r['mu'] * 1e3, 2)) for r in R.read_jsonl(p) if r['dropped']]
    if d:
        inv['/'.join(Path(p).parts[-4:-3] + Path(p).parts[-2:]).replace('calib_file-v2_eps_a-0_', '')] = d
L = ['', '## Inventory of NEW F5 / F5P drops (episode, mu x 1e-3); runs with zero drops omitted', '']
for k, v in inv.items():
    L.append(f'- {k}: {len(v)} drops, mu x1e-3 = {[m for _, m in v]}')
L.append(f'- Largest mu among all listed drops: {max(m for v in inv.values() for _, m in v)} x1e-3 (bin edges of the study: 0.5, 0.82, 1.4, 2.2, 3.7, 6.0 x1e-3).')
md = RB / 'analysis/RESULT_REBASELINE.md'
t = md.read_text()
t = t.split('\n## Inventory of NEW')[0].rstrip('\n') + '\n' + '\n'.join(L) + '\n'
md.write_text(t)
sj = RB / 'analysis/summary_rebaseline.json'
s = json.load(open(sj)); s['drop_inventory'] = inv; sj.write_text(json.dumps(s, indent=1))
print('\n'.join(L))
