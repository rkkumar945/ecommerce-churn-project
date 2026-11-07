"""README ke numbers ek jagah, source file ke saath.   python scripts/readme_facts.py"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def show(title, body):
    print(f'\n== {title}')
    print(body)


manifest = json.loads((ROOT / 'app_artifacts' / 'model_manifest.json').read_text())
show('Model (app_artifacts/model_manifest.json)', {
    'model': manifest['model'], 'params': manifest['params'], 'threshold': manifest['threshold'],
    'test_auc': round(manifest['test_auc'], 3), 'n_model_columns': len(manifest['features']),
    'built': manifest['created_at_utc'][:10], 'python': manifest['env']['python'],
})

table = pd.read_csv(ROOT / 'app_artifacts' / 'customer_features_app.csv')
te = table[table['split'] == 'test']
show('Customers (app_artifacts/customer_features_app.csv)', {
    'customers': len(table), 'train': int((table['split'] == 'train').sum()), 'held_out': len(te),
    'held_out_churners': int(te['observed_churn'].sum()),
    'churn_rate_overall': round(float(table['observed_churn'].mean()), 3),
})

for name in ('cv_ensembles_day83.csv', 'nested_cv_day84.csv', 'shap_group_importance_day85b.csv'):
    p = ROOT / 'reports' / name
    show(f'reports/{name}', pd.read_csv(p).round(3).to_string(index=False) if p.exists() else 'MISSING')

p = ROOT / 'reports' / 'model_comparison_report.md'
show('reports/model_comparison_report.md', p.read_text(encoding='utf-8') if p.exists() else 'MISSING')

lines = subprocess.run([sys.executable, '-m', 'pytest', '--collect-only', '-q'], cwd=ROOT,
                       capture_output=True, text=True).stdout.strip().splitlines()
show('Automated tests', lines[-1] if lines else 'pytest output empty')

commits = subprocess.run(['git', 'rev-list', '--count', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
show('Repo', {'commits': commits, 'notebooks': len(list((ROOT / 'notebooks').rglob('*.ipynb')))})