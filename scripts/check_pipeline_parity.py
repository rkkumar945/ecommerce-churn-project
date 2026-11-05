import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])   # hamesha repo root se chalao, chahe kahin se bulao


from pathlib import Path
import pandas as pd

# 1) Sabse naya run (naam se nahi, modification time se)
runs = sorted(Path('pipeline_runs').glob('asof-latest_*'), key=lambda p: p.stat().st_mtime)
if not runs:
    raise FileNotFoundError("pipeline_runs/ mein 'asof-latest_*' folder nahi mila (cwd repo root hona chahiye)")
run = runs[-1]
print('using run:', run.name)

# 2) Features: pipeline output vs purana v2 dataset
new = pd.read_csv(run / 'features.csv').sort_values('Customer ID').reset_index(drop=True)
old = pd.read_csv('data/processed/customer_ml_dataset_v2.csv').sort_values('Customer ID').reset_index(drop=True)
print(new.shape, old.shape)
only_new, only_old = set(new.columns) - set(old.columns), set(old.columns) - set(new.columns)
assert not only_new and not only_old, f'columns alag: sirf new={only_new}, sirf old={only_old}'
pd.testing.assert_frame_equal(new, old[new.columns], check_dtype=False, atol=1e-6)   # column order ignore, values strict

# 3) Holdout wahi hai? (split.csv mein label 'holdout' ya 'test' ho sakta hai)
app = pd.read_csv('app_artifacts/customer_features_app.csv')
sp = pd.read_csv(run / 'split.csv')
print('split labels in split.csv:', sorted(sp['split'].unique()))
label = 'holdout' if 'holdout' in set(sp['split']) else 'test'
a_ids = set(app.loc[app['split'] == 'test', 'Customer ID'])
s_ids = set(sp.loc[sp['split'] == label, 'Customer ID'])
assert a_ids and s_ids, f"holdout ids khaali: app={len(a_ids)}, split.csv('{label}')={len(s_ids)}"
assert a_ids == s_ids, f'holdout alag: sirf app={len(a_ids - s_ids)}, sirf split.csv={len(s_ids - a_ids)}'
print('holdout same: True', f'({len(a_ids)} customers)')

print((run / 'run_report.md').read_text())