"""Evidently drift report (HTML). Alag virtual env mein chalao: sirf pandas + evidently chahiye.
    python scripts/evidently_report.py --ref scratch/_drift_ref.csv --cur scratch/_drift_cur.csv --out reports/evidently_drift.html
"""
import sys
import pandas as pd
import pandas.core.dtypes.common
if not hasattr(pd.core.dtypes.common, "infer_dtype_from_object"):
    def _dummy_infer(val):
        return None
    pd.core.dtypes.common.infer_dtype_from_object = _dummy_infer  # evidently <0.7 mein ye chahiye, pandas >=2.0 mein hata diya

from pathlib import Path
import argparse
import json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))   # repo root ke src ko import path mein daalo


import evidently




def new_api(ref, cur, cols):                       # evidently >= 0.7
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset
    dd = DataDefinition(numerical_columns=cols)
    snap = Report([DataDriftPreset()]).run(Dataset.from_pandas(cur, data_definition=dd),
                                           Dataset.from_pandas(ref, data_definition=dd))
    return snap, snap.dict()


def legacy_namespace(ref, cur, cols):              # purana API, naye package ke andar
    from evidently.legacy.metric_preset import DataDriftPreset
    from evidently.legacy.pipeline.column_mapping import ColumnMapping
    from evidently.legacy.report import Report
    rep = Report(metrics=[DataDriftPreset()])
    rep.run(reference_data=ref, current_data=cur, column_mapping=ColumnMapping(numerical_features=cols))
    return rep, rep.as_dict()


def old_api(ref, cur, cols):                       # evidently <= 0.6
    from evidently import ColumnMapping
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report
    rep = Report(metrics=[DataDriftPreset()])
    rep.run(reference_data=ref, current_data=cur, column_mapping=ColumnMapping(numerical_features=cols))
    return rep, rep.as_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
    "--ref",
    default="scratch/_drift_ref.csv",
    help="Path to reference CSV",
)
    ap.add_argument(
    "--cur",
    default="scratch/_drift_cur.csv",
    help="Path to current CSV",
)
    ap.add_argument(
    "--out",
    default="reports/evidently_drift.html",
)
    a = ap.parse_args()

    import evidently
    print('evidently', getattr(evidently, '__version__', 'unknown'))
    ref, cur = pd.read_csv(a.ref), pd.read_csv(a.cur)
    cols = [c for c in ref.columns if c in cur.columns]

    result, last_err = None, None
    for fn in (new_api, legacy_namespace, old_api):
        try:
            result = fn(ref[cols], cur[cols], cols)
            print('used:', fn.__name__)
            break
        except ImportError as e:
            last_err = e
    if result is None:
        raise SystemExit(f'Koi Evidently API match nahi hua. Last error: {last_err!r}')

    report, as_dict = result
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out))
    out.with_suffix('.json').write_text(json.dumps(as_dict, default=str, indent=2))
    print('saved', out, 'and', out.with_suffix('.json'))


if __name__ == '__main__':
    main()