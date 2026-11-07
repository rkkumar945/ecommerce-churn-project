"""Repo audit (Day 98): mock review ke mechanical checks.   python scripts/repo_audit.py
FAIL ho to exit code 1. WARN/INFO sirf jaankari hain: har ek ko padho aur tay karo."""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {'.git', '__pycache__', 'node_modules', 'mlflow_store', 'pipeline_runs', 'scratch',
             '.ruff_cache', '.pytest_cache'}
EXPECTED_NB_DIRS = {'month1_sql', 'month2_python_pandas', 'month3_statistics_eda',
                    'month4_machine_learning', 'month5_advanced_mlops', 'weekly_tests'}
FORBIDDEN = [
    (r'(^|/)\.env$', 'secrets file'),
    (r'\.(pkl|pickle|joblib)$', 'model/pickle file (sirf app_artifacts/ mein allowed)'),
    (r'^data/(raw|processed|db)/(?!\.gitkeep)', 'data file'),
    (r'^(mlflow_store|mlruns|pipeline_runs|scratch)/', 'local run artifacts'),
    (r'__pycache__/|\.ipynb_checkpoints/|\.(pytest|ruff)_cache/', 'cache'),
    (r'(^|/)\.venv', 'virtualenv'),
]
LOCAL_PATH = re.compile(r'[A-Za-z]:\\Users\\|/Users/[^/\s]+|/home/[a-z]')
SECRET = re.compile(r'(api[_-]?key|secret|passw(or)?d|token)\s*[:=]\s*[\'"][^\'"\s]{8,}', re.IGNORECASE)


def git_files():
    out = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [f for f in out.split('\0') if f]


def read(rel):
    return (ROOT / rel).read_text(encoding='utf-8', errors='ignore')


def strip_fences(text):
    return re.sub(r'```.*?```', '', text, flags=re.DOTALL)


def exists_exact(base: Path, rel: str) -> bool:
    """Case-sensitive check, jaisa GitHub (Linux) karta hai."""
    p = base
    for part in Path(rel).parts:
        if part == '..':
            p = p.parent
            continue
        if part not in {c.name for c in p.iterdir()}:
            return False
        p = p / part
    return True


def walk_dirs():
    for dirpath, dirnames, _ in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith('.venv')]
        yield Path(dirpath)


def output_texts(cell):
    for o in cell.get('outputs', []):
        if 'text' in o:
            yield ''.join(o['text'])
        for k in ('text/plain', 'text/html'):
            v = (o.get('data') or {}).get(k)
            if v is not None:
                yield ''.join(v) if isinstance(v, list) else str(v)
        if o.get('output_type') == 'error':
            yield o.get('evalue', '')


def parse_req(path: Path) -> dict:
    pins = {}
    for raw in path.read_text().splitlines():
        line = raw.split('#')[0].strip()
        if not line:
            continue
        if line.startswith('-r'):
            pins.update(parse_req(path.parent / line[2:].strip()))
            continue
        if line.startswith('-'):
            continue
        m = re.match(r'^([A-Za-z0-9_.\-]+)\s*(?:==\s*([^\s;]+))?', line)
        pins[m.group(1).lower().replace('_', '-')] = m.group(2)
    return pins


def run_audit():
    F = []

    def add(sev, area, msg):
        F.append((sev, area, msg))

    files = git_files()

    # 1) tracked files jo kabhi nahi hone chahiye
    for f in files:
        for pat, label in FORBIDDEN:
            if re.search(pat, f) and not (f.startswith('app_artifacts/') and f.endswith('.joblib')):
                add('FAIL', 'tracked', f'{f}: {label}')
                break

    # 2) bade files
    for f in files:
        p = ROOT / f
        if p.exists() and not f.startswith('app_artifacts/'):
            mb = p.stat().st_size / 1e6
            if mb > 5 or (mb > 2 and f.endswith('.ipynb')):
                add('WARN', 'size', f'{f}: {mb:.1f} MB')

    # 3) stray files aur anjaan folders
    for f in files:
        n = Path(f).name.lower()
        if re.match(r'untitled|.*copy\d*\.|.*-copy', n) or n.endswith(('.bak', '.tmp', '~')):
            add('FAIL', 'stray', f)
    nb_root = ROOT / 'notebooks'
    if nb_root.exists():
        for d in sorted(p.name for p in nb_root.iterdir() if p.is_dir() and not p.name.startswith('.')):
            if d not in EXPECTED_NB_DIRS:
                add('WARN', 'structure', f'notebooks/{d}: unexpected folder')
    for d in walk_dirs():
        if d.name.lower() in {'model', 'models'} and d != ROOT / 'models':
            n_files = sum(1 for x in d.iterdir() if x.is_file())
            if n_files:
                add('WARN', 'structure', f'{d.relative_to(ROOT)}: nested model folder ({n_files} files)')

    # 4) notebooks
    for f in (x for x in files if x.endswith('.ipynb')):
        try:
            nb = json.loads(read(f))
        except json.JSONDecodeError:
            add('FAIL', 'notebook', f'{f}: invalid JSON')
            continue
        cells = nb.get('cells', [])
        counts = [c['execution_count'] for c in cells if c.get('cell_type') == 'code' and c.get('execution_count')]
        if counts and (counts != sorted(counts) or len(set(counts)) != len(counts)):
            add('WARN', 'notebook', f'{f}: execution order bikhra (stale outputs ho sakte hain)')
        elif counts and counts[0] != 1:
            add('WARN', 'notebook', f'{f}: Restart + Run All nahi hua (counter {counts[0]} se shuru)')
        n_err = sum(1 for c in cells for o in c.get('outputs', []) if o.get('output_type') == 'error')
        if n_err:
            add('WARN', 'notebook', f'{f}: {n_err} error output saved')
        hit_path = hit_secret = False
        for c in cells:
            for t in [''.join(c.get('source', [])), *output_texts(c)]:
                hit_path = hit_path or bool(LOCAL_PATH.search(t))
                hit_secret = hit_secret or bool(SECRET.search(t))
        if hit_path:
            add('WARN', 'privacy', f'{f}: local path (C:\\Users... ya /home/...) source ya output mein')
        if hit_secret:
            add('FAIL', 'secret', f'{f}: secret jaisa pattern')
        text = read(f)
        if not re.search(r'src\.paths|os\.chdir|find_root|\.\./\.\./', text) and re.search(r"['\"](data|models|reports)/", text):
            add('INFO', 'paths', f'{f}: repo-root relative paths, working directory par depend karta hai')

    # 5) duplicate split logic
    for f in files:
        if f.startswith('src/') and f.endswith('.py') and f != 'src/features/split.py' and 'train_test_split(' in read(f):
            add('WARN', 'duplication', f'{f}: train_test_split direct (make_split use karo)')
    n_nb = sum(1 for f in files if f.endswith('.ipynb') and 'train_test_split(' in read(f))
    add('INFO', 'duplication', f'{n_nb} notebooks mein train_test_split (history ke taur par theek)')

    # 6) TODO / FIXME
    skip_todo = {'scripts/repo_audit.py', 'tests/test_readme.py'}
    for f in files:
        if f.endswith(('.py', '.md')) and not f.startswith('notebooks/') and f not in skip_todo:
            for i, line in enumerate(read(f).splitlines(), 1):
                if re.search(r'\b(TODO|FIXME|XXX)\b', line):
                    add('WARN', 'todo', f'{f}:{i}')

    # 7) markdown links (exact case)
    for md in [f for f in files if f.endswith('.md') and not f.startswith('notebooks/')]:
        base = (ROOT / md).parent
        prose = strip_fences(read(md))
        targets = re.findall(r'!?\[[^\]]*\]\(([^)\s]+)', prose) + re.findall(r'(?:src|href)="([^"]+)"', prose)
        for t in targets:
            if t.startswith(('http://', 'https://', 'mailto:', '#')):
                continue
            path = t.split('#')[0]
            if path and not exists_exact(base, path):
                add('FAIL', 'links', f'{md}: broken link {t}')

    # 8) images jo kisi markdown mein nahi hain
    md_text = ' '.join(read(f) for f in files if f.endswith('.md'))
    for f in files:
        if (
        f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))
        and f.startswith(('docs/', 'reports/screenshots/'))
        and Path(f).name not in md_text
    ):
         add('WARN', 'images', f'{f}: kisi markdown mein reference nahi')

    # 9) requirements
    root_req, app_req = ROOT / 'requirements.txt', ROOT / 'src' / 'app' / 'requirements.txt'
    if root_req.exists() and app_req.exists():
        root, app = parse_req(root_req), parse_req(app_req)
        for lib, v in app.items():
            if lib not in root:
                add('FAIL', 'requirements', f'{lib}: app requirement root requirements.txt mein nahi')
            elif v and root[lib] and root[lib] != v:
                add('FAIL', 'requirements', f'{lib}: app {v} vs root {root[lib]}')
        for lib in sorted({'pytest', 'ruff', 'mlflow', 'scipy', 'statsmodels', 'imbalanced-learn', 'seaborn',
                           'requests', 'tabulate', 'pillow'} - set(root)):
            add('WARN', 'requirements', f'{lib}: root requirements.txt mein nahi (fresh clone par notebooks/tests fail honge)')
        if not ({'jupyter', 'notebook', 'ipykernel', 'jupyterlab'} & set(root)):
            add('WARN', 'requirements', 'ipykernel/jupyter root requirements.txt mein nahi')

    # 10) app artifacts
    art = ROOT / 'app_artifacts'
    need = {'final_model.joblib', 'model_manifest.json', 'customer_features_app.csv'}
    tracked = {Path(f).name for f in files if f.startswith('app_artifacts/')}
    if need - tracked:
        add('FAIL', 'artifacts', f'tracked nahi: {sorted(need - tracked)}')
    mf = art / 'model_manifest.json'
    if mf.exists() and (art / 'final_model.joblib').exists():
        m = json.loads(mf.read_text())
        if hashlib.sha256((art / 'final_model.joblib').read_bytes()).hexdigest() != m['model_file']['sha256']:
            add('FAIL', 'artifacts', 'model sha256 manifest se match nahi karta')
        report = ROOT / 'reports' / 'model_comparison_report.md'
        if report.exists() and f"{m['test_auc']:.3f}" not in report.read_text(encoding='utf-8'):
            add('WARN', 'consistency', f"model_comparison_report.md mein test AUC {m['test_auc']:.3f} nahi mila")
    return F


def main() -> int:
    findings = run_audit()
    order = {'FAIL': 0, 'WARN': 1, 'INFO': 2}
    for sev, area, msg in sorted(findings, key=lambda x: (order[x[0]], x[1], x[2])):
        print(f'{sev:5} {area:13} {msg}')
    counts = {s: sum(1 for f in findings if f[0] == s) for s in order}
    print('\nSummary:', counts)
    return 1 if counts['FAIL'] else 0


if __name__ == '__main__':
    sys.exit(main())