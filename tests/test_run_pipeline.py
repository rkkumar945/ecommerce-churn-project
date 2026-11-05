import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.pipeline.run_pipeline import MANIFEST, Config, build_dag, run_dir_for

pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason='champion recipe (app_artifacts manifest) chahiye')


def quiet(*_):
    pass


def make_raw(path, n_customers=600, seed=0):
    rng = np.random.default_rng(seed)
    start, span = pd.Timestamp('2010-01-01'), 707
    rows, inv = [], 500000
    for cid in range(10000, 10000 + n_customers):
        first = int(rng.integers(0, int(span * 0.7)))
        days = np.sort(rng.integers(first, span, int(rng.integers(1, 8))))
        country = str(rng.choice(['United Kingdom', 'Germany', 'France']))
        for d in days:
            inv += 1
            when = (start + pd.Timedelta(days=int(d), hours=9)).strftime('%Y-%m-%d %H:%M:%S')
            for _ in range(int(rng.integers(1, 4))):
                rows.append((str(inv), f'S{int(rng.integers(1, 40))}', 'ITEM', int(rng.integers(1, 10)),
                             when, round(float(rng.uniform(1, 20)), 2), float(cid), country))
    rows.append(('C999999', 'S1', 'ITEM', -1, '2011-03-01 10:00:00', 5.0, 10001.0, 'Germany'))   # cancellation
    rows.append(('600001', 'S2', 'ITEM', 2, '2011-03-02 10:00:00', 5.0, None, 'Germany'))        # guest checkout
    rows.append(('600002', 'S3', 'ITEM', 1, '2011-12-09 09:00:00', 5.0, 10002.0, 'Germany'))     # dataset ka aakhri din
    pd.DataFrame(rows, columns=['Invoice', 'StockCode', 'Description', 'Quantity', 'InvoiceDate',
                                'Price', 'Customer ID', 'Country']).to_csv(path, index=False)


def test_pipeline_end_to_end_cache_and_sealed_holdout(tmp_path):
    raw = tmp_path / 'raw.csv'
    make_raw(raw)
    cfg = Config(raw_csv=raw, run_root=tmp_path / 'runs')
    run_dir = run_dir_for(cfg)
    dag = build_dag(cfg, run_dir)
    manifest_hash = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()

    assert dag.plan()['clean'] == 'would_run'
    assert not run_dir.exists()                                     # dry-run ne kuch likha nahi

    first = dag.run(log=quiet)
    assert set(first.values()) == {'success'}, first
    assert dag.run(log=quiet) == {k: 'cached' for k in first}

    sp = pd.read_csv(run_dir / 'split.csv')
    assert set(sp['split']) == {'train', 'holdout'}
    info = json.loads((run_dir / 'challenger_cv.json').read_text())
    assert info['holdout_touched'] is False and info['n_train'] == int((sp['split'] == 'train').sum())
    assert (run_dir / 'run_report.md').exists() and (run_dir / 'challenger.joblib').exists()
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == manifest_hash     # app_artifacts nahi badla


def test_bad_raw_stops_pipeline_and_skips_downstream(tmp_path):
    raw = tmp_path / 'raw.csv'
    make_raw(raw)
    pd.read_csv(raw, dtype={'Invoice': str, 'StockCode': str}).drop(columns=['Price']).to_csv(raw, index=False)
    cfg = Config(raw_csv=raw, run_root=tmp_path / 'runs')
    run_dir = run_dir_for(cfg)

    status = build_dag(cfg, run_dir).run(log=quiet)
    assert status == {'clean': 'failed', 'features': 'skipped_upstream', 'split': 'skipped_upstream',
                      'challenger': 'skipped_upstream', 'report': 'skipped_upstream'}
    assert not (run_dir / 'challenger.joblib').exists()