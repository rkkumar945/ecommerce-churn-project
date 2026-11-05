"""Retraining pipeline (Day 94): clean -> features -> split -> challenger CV -> report.

Default = dry-run (sirf plan). --execute par asli run, aur sab kuch sirf pipeline_runs/<run_id>/ mein likhta hai.
data/processed aur app_artifacts ko yeh kabhi nahi chhoti. Holdout par kuch score nahi hota;
model selection aur promotion jaanbujhkar manual hain.

    python -m src.pipeline.run_pipeline                                  # dry-run
    python -m src.pipeline.run_pipeline --execute                        # asli run (data ki aakhri date tak)
    python -m src.pipeline.run_pipeline --as-of 2011-09-10 --execute     # backfill: purana snapshot
"""
import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # project root ko import path mein daalo

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.cleaning.clean_data import clean_transactions
from src.features.build_features import NUM_V2, build_customer_dataset, make_xy
from src.features.split import make_split
from src.models.search_spaces import get_search_spaces
from src.pipeline.checks import check_clean, check_features, check_raw, fail_on
from src.pipeline.mini_dag import DAG, Task

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'app_artifacts' / 'model_manifest.json'


@dataclass(frozen=True)
class Config:
    raw_csv: Path
    as_of: str | None = None
    horizon_days: int = 90
    cv_folds: int = 5
    seed: int = 42
    run_root: Path = ROOT / 'pipeline_runs'
    manifest: Path = MANIFEST
    track: bool = False


def run_dir_for(cfg: Config) -> Path:
    sha = hashlib.sha256(Path(cfg.raw_csv).read_bytes()).hexdigest()[:8]
    tag = re.sub(r'[^0-9A-Za-z-]+', '_', str(cfg.as_of)) if cfg.as_of else 'latest'   # ':' Windows folder naam mein nahi chalta
    return Path(cfg.run_root) / f'asof-{tag}_h{cfg.horizon_days}_{sha}'


def _atomic_csv(df: pd.DataFrame, path: Path):
    """Pehle temp file, phir rename: beech mein toot jaye to adhoori file output nahi banti (idempotent)."""
    tmp = path.with_name(path.name + '.tmp')
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _read_tx(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={'Invoice': str, 'StockCode': str}, parse_dates=['InvoiceDate'])


def build_dag(cfg: Config, run_dir: Path) -> DAG:
    p = {'cleaned': run_dir / 'cleaned.csv', 'features': run_dir / 'features.csv',
         'split': run_dir / 'split.csv', 'cv': run_dir / 'challenger_cv.json',
         'model': run_dir / 'challenger.joblib', 'report': run_dir / 'run_report.md'}
    cfg_tag = f'{cfg.as_of}|{cfg.horizon_days}|{cfg.seed}|{cfg.cv_folds}'     # config badla to cache invalid

    def clean():
        raw = pd.read_csv(cfg.raw_csv, dtype={'Invoice': str, 'StockCode': str})
        fail_on(check_raw(raw), 'raw')
        out = clean_transactions(raw)
        fail_on(check_clean(out), 'cleaned')
        _atomic_csv(out, p['cleaned'])

    def features():
        feat = build_customer_dataset(_read_tx(p['cleaned']), horizon_days=cfg.horizon_days, as_of=cfg.as_of)
        fail_on(check_features(feat), 'features')
        _atomic_csv(feat, p['features'])

    def split():
        feat = pd.read_csv(p['features'])
        _, idx_te = make_split(feat, seed=cfg.seed)
        out = pd.DataFrame({'Customer ID': feat['Customer ID'], 'split': 'train'})
        out.loc[idx_te, 'split'] = 'holdout'
        _atomic_csv(out, p['split'])

    def challenger():
        feat = pd.read_csv(p['features'])
        sp = pd.read_csv(p['split'])
        assert (sp['Customer ID'].to_numpy() == feat['Customer ID'].to_numpy()).all(), 'split aur features ki rows alag hain'
        train = feat[(sp['split'] == 'train').to_numpy()].reset_index(drop=True)      # holdout ko haath nahi lagaya

        recipe = json.loads(Path(cfg.manifest).read_text())
        family, params = recipe['model'], recipe['params']                           # sirf recipe padhi, test_auc nahi
        X = make_xy(train, NUM_V2).astype(float)
        y = train['Churned']
        if list(X.columns) != list(recipe['features']):
            raise ValueError('feature schema champion se alag hai: challenger comparable nahi hoga')

        est = clone(get_search_spaces(float((y == 0).sum() / (y == 1).sum()))[family][0]).set_params(**params)
        if 'n_jobs' in est.get_params():
            est.set_params(n_jobs=-1)

        cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
        scoring = {'roc_auc': 'roc_auc', 'ap': 'average_precision', 'recall': 'recall', 'f1': 'f1'}
        res = cross_validate(est, X, y, cv=cv, scoring=scoring)
        summary = {}
        for n in scoring:
            s = np.asarray(res[f'test_{n}'], dtype=float)          # sklearn ke 'test_*' = CV folds, holdout nahi
            summary[f'cv_{n}_mean'], summary[f'cv_{n}_std'] = float(s.mean()), float(s.std())

        joblib.dump(clone(est).fit(X, y), p['model'])
        p['cv'].write_text(json.dumps({
            'run_id': run_dir.name, 'family': family, 'params': params, 'n_train': len(X),
            'train_churn_rate': round(float(y.mean()), 4), 'cv': summary,
            'holdout_touched': False, 'sklearn': sklearn.__version__}, indent=2))

        if cfg.track:
            import mlflow

            from src.tracking.mlflow_utils import (
                git_info,
                log_cv_metrics,
                setup_tracking,
            )
            setup_tracking('churn-retrain')
            with mlflow.start_run(run_name=f'challenger_{run_dir.name}'):
                mlflow.log_params({'family': family, 'as_of': str(cfg.as_of), 'n_train': len(X)})
                mlflow.set_tags({**git_info(), 'stage': 'retrain_cv', 'test_set_used': 'false'})
                log_cv_metrics(summary)

    def report():
        info = json.loads(p['cv'].read_text())
        feat, sp, cv = pd.read_csv(p['features']), pd.read_csv(p['split']), info['cv']
        p['report'].write_text(f"""# Pipeline run: {run_dir.name}

- as_of: {cfg.as_of or 'latest date in data'} | horizon: {cfg.horizon_days} days
- customers: {len(feat):,} | churn rate: {feat['Churned'].mean():.1%}
- train: {int((sp['split'] == 'train').sum()):,} | holdout (sealed, not scored): {int((sp['split'] == 'holdout').sum()):,}
- recipe: {info['family']} (pre-registered, hyperparameters not re-tuned)

## Challenger CV (train only)
- ROC-AUC {cv['cv_roc_auc_mean']:.3f} +/- {cv['cv_roc_auc_std']:.3f}
- Average precision {cv['cv_ap_mean']:.3f} | recall {cv['cv_recall_mean']:.3f} | F1 {cv['cv_f1_mean']:.3f}

## Promotion checklist (MANUAL)
- [ ] All data checks passed (this run completed)
- [ ] CV AUC within noise of the champion (or better)
- [ ] Holdout evaluated once, outside this pipeline, on a time-based holdout
- [ ] Reviewer sign-off, then `python -m src.app.export_artifacts`
""")

    return DAG([
        Task('clean', clean, inputs=[cfg.raw_csv], outputs=[p['cleaned']], version=cfg_tag),
        Task('features', features, deps=['clean'], inputs=[p['cleaned']], outputs=[p['features']], version=cfg_tag),
        Task('split', split, deps=['features'], inputs=[p['features']], outputs=[p['split']], version=cfg_tag),
        Task('challenger', challenger, deps=['split'], inputs=[p['features'], p['split'], cfg.manifest],
             outputs=[p['cv'], p['model']], version=cfg_tag,
             retries=1 if cfg.track else 0, retry_delay=2.0),
        Task('report', report, deps=['challenger'], inputs=[p['cv']], outputs=[p['report']], version=cfg_tag),
    ], state_dir=run_dir / '.state')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Churn retraining pipeline (dry-run by default)')
    ap.add_argument('--raw', default=str(ROOT / 'data' / 'raw' / 'online_retail_II.csv'))
    ap.add_argument('--as-of', default=None, help='YYYY-MM-DD ya timestamp (default: data ki aakhri date)')
    ap.add_argument('--horizon-days', type=int, default=90)
    ap.add_argument('--execute', action='store_true', help='asli run (default: dry-run)')
    ap.add_argument('--force', action='store_true', help='cache ignore karo')
    ap.add_argument('--track', action='store_true', help='CV metrics MLflow mein bhi log karo')
    a = ap.parse_args(argv)

    cfg = Config(raw_csv=Path(a.raw), as_of=a.as_of, horizon_days=a.horizon_days, track=a.track)
    if not cfg.raw_csv.exists():
        print(f'raw file nahi mili: {cfg.raw_csv}')
        return 2
    if cfg.as_of is not None:
        pd.Timestamp(cfg.as_of)                                  # galat format yahin pakda jaaye

    run_dir = run_dir_for(cfg)
    dag = build_dag(cfg, run_dir)
    if not a.execute:
        print(f'Run dir : {run_dir}')
        print('Levels  :', dag.levels())
        return 0

    status = dag.run(force=a.force)
    print(status)
    return 0 if all(s in ('success', 'cached') for s in status.values()) else 1


if __name__ == '__main__':
    sys.exit(main())