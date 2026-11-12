"""Release gate (Day 100).   python scripts/release_check.py v1.0
Har gate PASS/FAIL. Koi FAIL ho to exit code 1: tag mat lagao."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE_URL = 'https://ecommerce-churn-project.streamlit.app/'
results = []


def sh(*args):
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True)


def gate(name, ok, detail=''):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


tag = sys.argv[1] if len(sys.argv) > 1 else 'v1.0'

branch = sh('git', 'rev-parse', '--abbrev-ref', 'HEAD').stdout.strip()
gate('on main', branch == 'main', f'(branch: {branch})')
gate('working tree clean', sh('git', 'status', '--porcelain').stdout.strip() == '', '(commit ya stash karo)')
sh('git', 'fetch')
counts = sh('git', 'rev-list', '--left-right', '--count', 'HEAD...origin/main').stdout.split()
ahead, behind = (int(counts[0]), int(counts[1])) if len(counts) == 2 else (-1, -1)
gate('in sync with origin/main', (ahead, behind) == (0, 0), f'(ahead {ahead}, behind {behind})')
gate('tag is new', sh('git', 'tag', '-l', tag).stdout.strip() == '', tag)
gate('LICENSE exists', any((ROOT / n).exists() for n in ('LICENSE', 'LICENSE.md', 'LICENSE.txt')))

art = ROOT / 'app_artifacts'
manifest = json.loads((art / 'model_manifest.json').read_text())
card = json.loads((ROOT / 'reports' / 'final_model_card.json').read_text())
sha = hashlib.sha256((art / 'final_model.joblib').read_bytes()).hexdigest()
gate('model sha256 matches manifest', sha == manifest['model_file']['sha256'])
gate('model card AUC matches manifest', abs(card['test_auc'] - manifest['test_auc']) < 1e-9)
gate('master test report exists', (ROOT / 'reports' / 'master_test_day100.md').exists())

gate('pytest', sh(sys.executable, '-m', 'pytest', '-q').returncode == 0)
gate('repo audit (0 FAIL)', sh(sys.executable, 'scripts/repo_audit.py').returncode == 0)

try:
    import requests
    code = requests.get(LIVE_URL, timeout=20).status_code
    print(f'INFO  live URL HTTP {code} (sleeping app par bhi 200 aa sakta hai, isliye 3 IDs khud check karo)')
except Exception as e:  # noqa: BLE001
    print(f'INFO  live URL check skipped: {e!r}')

print('\nALL GATES PASS' if all(results) else '\nGATES FAILED: tag mat lagao')
sys.exit(0 if all(results) else 1)