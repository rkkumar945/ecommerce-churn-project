"""Same-folds CV comparison helper."""
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_validate

SCORING = ['roc_auc', 'average_precision', 'f1', 'recall', 'precision']


def cv_compare(models: dict, X, y, cv, scoring=SCORING):
    """Har model ko wahi folds par chalao. (summary table, fold-wise AUC) return karta hai."""
    rows, folds = [], {}
    for name, model in models.items():
        res = cross_validate(clone(model), X, y, cv=cv, scoring=scoring,
                             return_train_score=True, n_jobs=1)   # models khud n_jobs=-1 use karte hain
        row = {'model': name}
        row.update({m: res[f'test_{m}'].mean() for m in scoring})
        row['auc_std'] = res['test_roc_auc'].std()
        row['train_auc'] = res['train_roc_auc'].mean()
        row['fit_sec'] = res['fit_time'].mean()
        rows.append(row)
        folds[name] = res['test_roc_auc']
    return pd.DataFrame(rows).set_index('model'), pd.DataFrame(folds)