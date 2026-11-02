import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app.service import (
    add_scores,
    apply_whatif,
    driver_frame,
    env_problems,
    find_customer,
    load_artifacts,
    load_table,
    pick_random,
    predict_row,
    readable_profile,
    top_k_summary,
    top_reasons,
    whatif_problems,
    wilson_interval,
)

st.set_page_config(page_title='Churn Risk Explorer', page_icon='📉', layout='wide')


@st.cache_resource
def get_data():
    model, manifest = load_artifacts()
    table = add_scores(model, load_table(), manifest['features'])
    background = table[manifest['features']].astype(float).sample(min(300, len(table)), random_state=0)
    return model, manifest, table, background.reset_index(drop=True)   # shared objects: mutate mat karna


st.title('Customer Churn Risk Explorer')
st.caption('Ranks customers by their risk of not ordering in the 90 days after the cutoff date.')

try:
    model, manifest, table, background = get_data()
except Exception as e: # noqa: BLE001
    st.error(f'App load nahi hui: {e}')
    st.stop()

FEATURES = manifest['features']
THRESHOLD = manifest['threshold']

for key, default in {'selected_id': None, 'history': [], 'wi_version': 0}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def clear_history():
    st.session_state.history = []


def reset_whatif():
    st.session_state.wi_version += 1


for p in env_problems(manifest):
    st.warning(f'Library version mismatch: {p}')

with st.sidebar:
    st.header('Model')
    st.write(f"**{manifest['model']}**")
    st.metric('Test ROC-AUC', f"{manifest['test_auc']:.3f}")
    st.metric('Alert threshold', f"{THRESHOLD:.2f}")
    st.caption('Scores are for ranking, not calibrated risk percentages.')
    st.caption(f"Built {manifest['created_at_utc'][:10]} · commit {manifest['git_commit']}")
    st.divider()
    st.subheader('Scored this session')
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), hide_index=True)
        st.button('Clear history', on_click=clear_history)
    else:
        st.caption('Nothing scored yet.')
    st.caption(f'{len(table):,} customers in the lookup table')


# ---------- cached computations ----------
@st.cache_data(show_spinner=False)
def drivers_for(customer_id: int):
    row = table.loc[table['Customer ID'] == customer_id].iloc[0]
    return driver_frame(model, row, FEATURES, background)


@st.cache_data(show_spinner=False)
def reasons_for(ids: tuple) -> dict:
    rows = table[table['Customer ID'].isin(ids)]
    return dict(zip(rows['Customer ID'].tolist(), top_reasons(model, rows, FEATURES, background)))


# ---------- charts (Figure(): pyplot ki global state nahi, Streamlit ke liye thread-safe) ----------
def _clean(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def score_position_chart(scores, score, threshold):
    fig = Figure(figsize=(6, 2.6))
    ax = fig.subplots()
    ax.hist(scores, bins=40, color='#c7c7c7')
    ax.axvline(threshold, color='#d62728', linestyle='--', label=f'alert threshold ({threshold:.2f})')
    ax.axvline(score, color='black', linewidth=2, label=f'this customer ({score:.2f})')
    ax.set_xlabel('Model score')
    ax.set_ylabel('Customers')
    ax.legend(fontsize=8, frameon=False)
    _clean(ax)
    fig.tight_layout()
    return fig


def drivers_chart(df, n=8):
    top = df.head(n).iloc[::-1]
    labels = [t if len(t) <= 46 else t[:45] + '…' for t in top['text']]
    fig = Figure(figsize=(6, 3.4))
    ax = fig.subplots()
    ax.barh(labels, top['shap'], color=['#d62728' if v > 0 else '#1f77b4' for v in top['shap']])
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Effect on score (model units). Right = higher risk')
    ax.tick_params(labelsize=8)
    _clean(ax)
    fig.tight_layout()
    return fig


# ---------- what-if (Day 89 jaisa hi) ----------
def slider_or_fixed(label, lo, hi, value, key, note=''):
    if hi <= lo:
        st.caption(f'{label}: **{value}** (fixed{note})')
        return value
    return st.slider(label, lo, hi, value, key=key)


def what_if_panel(row, base):
    cid, ver = int(row['Customer ID']), st.session_state.wi_version
    tenure, freq = int(row['Tenure']), int(row['Frequency'])
    single = freq == 1

    c1, c2, c3 = st.columns(3)
    with c1:
        rec = slider_or_fixed('Days since last order (at cutoff)', 0, 0 if single else tenure,
                              int(row['Recency']), f'wi_{cid}_{ver}_rec',
                              ' · one-order customers: equals customer age' if single else '')
    with c2:
        o90 = slider_or_fixed('Orders in the 90 days before cutoff', 0, freq,
                              int(row['Orders_last90']), f'wi_{cid}_{ver}_o90')
    with c3:
        p90 = slider_or_fixed('Orders in the 90 days before that', 0, freq,
                              int(row['Orders_prev90']), f'wi_{cid}_{ver}_p90')

    changes = {'Recency': rec, 'Orders_last90': o90, 'Orders_prev90': p90}
    changed = {k: v for k, v in changes.items() if v != int(row[k])}
    st.button('Reset sliders', on_click=reset_whatif)

    if not changed:
        st.caption('Move a slider to see how the score reacts.')
        return
    problems = whatif_problems(row, changes)
    if problems:
        for msg in problems:
            st.error(msg)
        return

    new_row = apply_whatif(row, changes)
    res = predict_row(model, new_row, FEATURES, THRESHOLD, table['score'])

    a, b, c = st.columns(3)
    a.metric('What-if score', f"{res['score']:.2f}", delta=f"{res['score'] - base['score']:+.2f}",
             delta_color='inverse')
    b.metric('What-if risk rank', f"Top {max(1, round(res['top_share'] * 100))}%")
    c.metric('What-if alert', 'FLAGGED' if res['flagged'] else 'OK')
    if res['flagged'] != base['flagged']:
        st.warning('The alert status flips in this scenario.')

    shown = []
    for f in list(changed) + ['OverdueRatio', 'OrderTrend', 'Spend_last90_log']:
        before, after = float(row[f]), float(new_row[f])
        if not np.isclose(before, after):
            shown.append({'feature': f, 'before': round(before, 3), 'after': round(after, 3)})
    st.caption('Features that changed (the last ones are recomputed automatically):')
    st.dataframe(pd.DataFrame(shown), hide_index=True)
    st.caption('Sensitivity only: every other feature is held fixed. This shows how the model reacts, '
               'not what would happen to a real customer.')


# ---------- customer view ----------
def show_customer(row):
    cid = int(row['Customer ID'])
    base = predict_row(model, row, FEATURES, THRESHOLD, table['score'])
    st.subheader(f"Customer {cid} · {row['Country']}")

    if base['flagged']:
        st.warning(f"⚠️ **FLAGGED**: score {base['score']:.2f} is at or above the alert threshold "
                   f"({THRESHOLD:.2f}). Prioritise for retention outreach.")
    else:
        st.success(f"🟢 **Not flagged**: score {base['score']:.2f} is below the alert threshold "
                   f"({THRESHOLD:.2f}).")

    left, right = st.columns(2)
    with left:
        m1, m2 = st.columns(2)
        m1.metric('Model score', f"{base['score']:.2f}",
                  help='Ranking score between 0 and 1. Not a calibrated probability.')
        m2.metric('Risk rank', f"Top {max(1, round(base['top_share'] * 100))}%",
                  help='Share of customers with an equal or higher score.')
        st.pyplot(score_position_chart(table['score'], base['score'], THRESHOLD))
        st.caption('Grey: all customers. Black line: this customer. Red dashed line: alert threshold.')

    with right:
        st.markdown('**Why this score?**')
        try:
            d, space = drivers_for(cid)
        except Exception as e: # noqa: BLE001
            st.caption(f'Explanation unavailable: {e}')
        else:
            st.pyplot(drivers_chart(d))
            ups, downs = d[d['shap'] > 0].head(3), d[d['shap'] < 0].head(3)
            u, w = st.columns(2)
            u.markdown('**Pushing the score up**')
            for _, r in ups.iterrows():
                u.markdown(f"- {r['text']}")
            if ups.empty:
                u.caption('None.')
            w.markdown('**Pulling the score down**')
            for _, r in downs.iterrows():
                w.markdown(f"- {r['text']}")
            if downs.empty:
                w.caption('None.')
            st.caption(f'Contributions are in the model\'s raw units ({space}), so use them to order the '
                       'reasons, not as percentages. They show what the model uses, not what causes churn.')

    if row['split'] == 'train':
        st.info('The model was **trained on this customer**, so the score here looks better than it would for '
                'a brand-new customer. Pick a test-split customer to see honest behaviour.')
    else:
        st.success('This customer was **held out** of training (test split).')

    outcome = 'did not place another order' if row['observed_churn'] == 1 else 'placed another order'
    st.write(f'**Observed in the data:** in the 90 days after the cutoff this customer {outcome}.')

    st.markdown('#### What-if (sensitivity)')
    what_if_panel(row, base)

    st.markdown('#### Customer profile (what the model saw, before the cutoff)')
    items = list(readable_profile(row).items())
    for start in range(0, len(items), 5):
        cols = st.columns(5)
        for col, (name, value) in zip(cols, items[start:start + 5]):
            col.metric(name, value)
    st.caption('The 90-day windows are measured backwards from the cutoff date.')

    with st.expander(f'All {len(FEATURES)} model inputs (exact values scored)'):
        st.dataframe(pd.DataFrame({'feature': FEATURES, 'value': [float(row[f]) for f in FEATURES]}),
                     hide_index=True)


MODES = ['Type a Customer ID', 'Random customer', 'Random flagged customer']

tab_customer, tab_list, tab_about = st.tabs(['Customer', 'Top at-risk list', 'About this model'])

# ================= Tab 1: customer =================
with tab_customer:
    with st.form('lookup'):
        mode = st.selectbox('How to pick a customer', MODES)
        id_text = st.text_input('Customer ID', placeholder='e.g. 12347',
                                help='Used only when "Type a Customer ID" is selected.')
        submitted = st.form_submit_button('Score customer')

    if submitted:
        if mode == MODES[0]:
            picked, err = find_customer(table, id_text)
        else:
            picked = pick_random(table, flagged_only=(mode == MODES[2]), threshold=THRESHOLD)
            err = None if picked is not None else 'Koi customer match nahi hua.'
        if err:
            st.error(err)
        else:
            st.session_state.selected_id = int(picked['Customer ID'])
            entry = {'Customer ID': int(picked['Customer ID']), 'Score': round(float(picked['score']), 2),
                     'Flagged': bool(picked['score'] >= THRESHOLD)}
            st.session_state.history = ([entry] + st.session_state.history)[:10]

    sid = st.session_state.selected_id
    if sid is None:
        st.info('Pick a customer above and press **Score customer**.')
    else:
        show_customer(table.loc[table['Customer ID'] == sid].iloc[0])

# ================= Tab 2: top at-risk list =================
with tab_list:
    st.markdown('Rank customers by model score and choose how much of the list your retention budget can cover.')
    c1, c2 = st.columns(2)
    budget = c1.slider('Budget: % of customers to contact', 1, 30, 10)
    held_out = c2.checkbox('Held-out (test) customers only', value=True,
                           help='Training customers have optimistic scores, so the honest view uses test customers only.')

    pool = (table[table['split'] == 'test'] if held_out else table).reset_index(drop=True)
    s = top_k_summary(pool, budget / 100)
    lo, hi = wilson_interval(round(s['precision'] * s['k']), s['k'])

    if not held_out:
        st.warning('This view includes customers the model was trained on, so the numbers below are optimistic.')

    m = st.columns(4)
    m[0].metric('Customers contacted', f"{s['k']:,}", help=f"out of {s['n']:,} in this view")
    m[1].metric('Actually did not reorder', f"{s['precision']:.0%}",
                help=f'95% interval {lo:.0%} to {hi:.0%}. Base rate in this view: {s["base_rate"]:.0%}.')
    m[2].metric('Lift vs random', f"{s['lift']:.1f}x", help='Precision divided by the base rate.')
    m[3].metric('Churners reached', f"{s['recall']:.0%}",
                help='Share of all observed non-returners in this view that are on the list.')
    st.caption(f'Interval for the precision: {lo:.0%} to {hi:.0%}. With few customers this interval is wide, '
               'so do not read small differences between budgets as real.')

    ranked = pool.sort_values('score', ascending=False).head(s['k'])
    view = pd.DataFrame({
        'Position': np.arange(1, len(ranked) + 1),
        'Customer ID': ranked['Customer ID'].astype(int).to_numpy(),
        'Country': ranked['Country'].to_numpy(),
        'Score': ranked['score'].round(3).to_numpy(),
        'Flagged': np.where(ranked['score'] >= THRESHOLD, 'Yes', 'No'),
    })

    if st.checkbox('Add main reason per customer (slower)', value=False):
        ids = tuple(view['Customer ID'].head(300).tolist())
        with st.spinner('Computing reasons...'):
            mapping = reasons_for(ids)
        view['Main reason'] = view['Customer ID'].map(mapping).fillna('-')
        if len(view) > 300:
            st.caption('Reasons are computed for the first 300 customers only.')

    st.caption(f"{int((view['Flagged'] == 'Yes').sum())} of the {len(view)} customers on this list are above "
               'the alert threshold; the rest are included only because of the budget you chose.')
    st.dataframe(view.head(500), hide_index=True)
    st.download_button('Download this list (CSV)', data=view.to_csv(index=False).encode('utf-8'),
                       file_name=f'at_risk_top_{budget}pct.csv', mime='text/csv')
    st.caption('The file has no outcome columns: a real campaign list would not know the outcome in advance.')

# ================= Tab 3: about =================
with tab_about:
    te = table[table['split'] == 'test']
    st.markdown(f"""
**What the model predicts.** Whether a customer places *no order* in the 90 days after a cutoff date
(90 days before the last date in the data). Features use only orders **before** the cutoff, so the model
predicts the future instead of re-learning its own label.

**Model.** {manifest['model']}, chosen using cross-validation only and recorded before the test set was scored.
Test ROC-AUC: **{manifest['test_auc']:.3f}** on {len(te):,} held-out customers
({int(te['observed_churn'].sum()):,} of them did not reorder).

**How to read the score.** It is a *ranking* score. Class weighting inflates raw values, so a score of 0.7 does
**not** mean a 70% chance. Use rank and the alert flag instead. The alert threshold ({THRESHOLD:.2f}) was picked
on out-of-fold predictions using assumed costs (missing a churner is costly, a wasted offer is cheap).

**Limits.**
- One UK online retailer, Dec 2009 to Dec 2011, so patterns may not transfer to other businesses.
- The 90-day churn window is a business assumption.
- Explanations show what the model uses, not what causes churn. The what-if sliders are sensitivity checks.
- The test set is small, so every number has a wide uncertainty band.
""")
    with st.expander('Build details'):
        st.json({'git_commit': manifest['git_commit'], 'built_utc': manifest['created_at_utc'],
                 'serialization': manifest['serialization'], 'env': manifest['env']})