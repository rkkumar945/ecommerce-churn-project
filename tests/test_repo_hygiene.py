import importlib.util
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location('repo_audit', ROOT / 'scripts' / 'repo_audit.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(shutil.which('git') is None or not (ROOT / '.git').exists(), reason='needs a git checkout')
def test_no_fail_level_findings():
    fails = [f for f in _load().run_audit() if f[0] == 'FAIL']
    assert not fails, '\n'.join(f'{area}: {msg}' for _, area, msg in fails)