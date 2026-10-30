"""SHAP helpers: model-type ke hisaab se explainer, churn (class 1) ke values return."""
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def _positive_class(sv):
    """Alag shap versions alag shape deti hain. Class 1 ke liye (n, features) array return karo."""
    if isinstance(sv, list):
        return np.asarray(sv[1])
    sv = np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv


def _base_value(ev):
    ev = np.atleast_1d(np.asarray(ev, dtype=float))
    return float(ev[-1])


def _shap_for_estimator(est, Xt, bg_t):
    if hasattr(est, 'coef_'):                                  # linear model
        ex = shap.LinearExplainer(est, bg_t)
        values = _positive_class(ex.shap_values(Xt))
        output, space = est.decision_function(Xt), 'log-odds'
    else:                                                      # tree-based
        ex = shap.TreeExplainer(est)
        values = _positive_class(ex.shap_values(Xt))
        if type(est).__name__ == 'XGBClassifier':
            output, space = est.predict(Xt, output_margin=True), 'log-odds'
        else:
            output, space = est.predict_proba(Xt)[:, 1], 'probability'
    return values, _base_value(ex.expected_value), np.asarray(output, dtype=float), space


def explain(model, X: pd.DataFrame, background: pd.DataFrame | None = None) -> dict:
    """values (DataFrame, X ke hi columns/order mein), base, output, space, gap return karta hai.
    gap = max |sum(values) + base - model ka asli output|. Ye ~0 hona chahiye."""
    X = X.reset_index(drop=True)
    if isinstance(model, Pipeline):
        prep, est = model[:-1], model[-1]
        names = [n.split('__')[-1] for n in prep.get_feature_names_out()]
        Xt = pd.DataFrame(prep.transform(X), columns=names)
        bg_t = pd.DataFrame(prep.transform(X if background is None else background), columns=names)
    else:
        est, Xt = model, X
        bg_t = X if background is None else background

    values, base, output, space = _shap_for_estimator(est, Xt, bg_t)
    values = pd.DataFrame(values, columns=Xt.columns)[list(X.columns)]   # ColumnTransformer ne order badla ho to wapas theek
    gap = float(np.abs(values.sum(axis=1).to_numpy() + base - output).max())
    return {'values': values, 'base': base, 'output': output, 'space': space, 'gap': gap}


def reason_codes(values_row, x_row: pd.Series, k: int = 3):
    """Top-k features jo is customer ka churn risk badhate hain (sirf positive SHAP): [(feature, value, shap)]."""
    s = pd.Series(np.asarray(values_row), index=x_row.index)
    top = s[s > 0].sort_values(ascending=False).head(k)
    return [(f, float(x_row[f]), float(v)) for f, v in top.items()]