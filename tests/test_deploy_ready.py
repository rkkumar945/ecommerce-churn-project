import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_REQ = ROOT / 'src' / 'app' / 'requirements.txt'
MANIFEST = ROOT / 'app_artifacts' / 'model_manifest.json'
DEV_ONLY = {'mlflow', 'evidently', 'jupyter', 'notebook', 'ipykernel', 'pytest', 'ruff',
            'statsmodels', 'prefect', 'apache-airflow', 'tabulate', 'nbstripout'}
RUNTIME_NEEDS = {'streamlit', 'pandas', 'numpy', 'scikit-learn', 'joblib', 'matplotlib', 'shap'}


def parse_requirements(path: Path) -> dict:
    pins = {}
    for raw in path.read_text().splitlines():
        line = raw.split('#')[0].strip()
        if not line or line.startswith('-'):
            continue
        m = re.match(r'^([A-Za-z0-9_.\-]+)\s*(?:==\s*([^\s;]+))?', line)
        pins[m.group(1).lower().replace('_', '-')] = m.group(2)
    return pins


def test_entrypoint_and_dependency_file_live_together():
    assert (ROOT / 'src' / 'app' / 'app.py').exists()
    assert APP_REQ.exists(), 'Cloud entrypoint ki directory mein requirements.txt dhoondhta hai'
    assert not (ROOT / 'requirements-app.txt').exists(), 'purana naam hata do: Cloud ise nahi padhta'


def test_pins_are_exact_and_runtime_complete():
    pins = parse_requirements(APP_REQ)
    assert all(pins.values()), f'unpinned: {[k for k, v in pins.items() if not v]}'
    assert RUNTIME_NEEDS <= set(pins), f'missing: {RUNTIME_NEEDS - set(pins)}'
    assert not (DEV_ONLY & set(pins)), f'dev-only packages: {DEV_ONLY & set(pins)}'


@pytest.mark.skipif(not MANIFEST.exists(), reason='app_artifacts missing')
def test_pins_match_the_versions_the_model_was_built_with():
    manifest = json.loads(MANIFEST.read_text())
    pins = parse_requirements(APP_REQ)
    for lib, built in manifest['env'].items():
        if lib in pins:
            assert pins[lib] == built, f'{lib}: requirements {pins[lib]} vs built {built}'
    if manifest['model'] == 'XGBoost':
        assert 'xgboost' in pins


@pytest.mark.skipif(not MANIFEST.exists(), reason='app_artifacts missing')
def test_artifacts_are_tracked_by_git():
    try:
        out = subprocess.run(['git', 'ls-files', 'app_artifacts'], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.split()
    except Exception: # noqa: BLE001
        pytest.skip('git not available')
    for name in ('final_model.joblib', 'model_manifest.json', 'customer_features_app.csv'):
        assert f'app_artifacts/{name}' in out, f'{name} git mein tracked nahi: Cloud ko file milegi hi nahi'


def test_runtime_env_reports_python_and_libs():
    from src.app.service import runtime_env
    env = runtime_env()
    assert env['python'].count('.') == 2 and 'numpy' in env