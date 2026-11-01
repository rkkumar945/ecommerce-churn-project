import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app.service import (
    add_scores,
    env_problems,
    find_customer,
    load_artifacts,
    load_table,
    pick_random,
    readable_profile,
)

st.set_page_config(page_title='Churn Risk Explorer', page_icon='📉', layout='wide')


@st.cache_resource
def get_data():
    model, manifest = load_artifacts()
    table = add_scores(model, load_table(), manifest['features'])
    return model, manifest, table          # shared objects: inhe mutate mat karna, .copy() lo


st.title('Customer Churn Risk Explorer')
st.caption('Ranks customers by their risk of not ordering in the 90 days after the cutoff date.')

try:
    model, manifest, table = get_data()
except (FileNotFoundError, KeyError, ValueError) as e: 
    st.error(f'App load nahi hui: {e}')
    st.stop()

THRESHOLD = manifest['threshold']

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
    st.caption(f'{len(table):,} customers in the lookup table')


def show_customer(row):
    flagged = row['score'] >= THRESHOLD
    st.subheader(f"Customer {int(row['Customer ID'])} · {row['Country']}")

    c1, c2, c3 = st.columns(3)
    c1.metric('Model score', f"{row['score']:.2f}",
              help='Ranking score between 0 and 1. Not a calibrated probability.')
    c2.metric('Risk rank', f"Top {max(1, round(row['top_share'] * 100))}%",
              help='Share of customers with an equal or higher score.')
    c3.metric('Alert', 'FLAGGED' if flagged else 'OK', help=f'Flagged when score >= {THRESHOLD:.2f}.')

    if row['split'] == 'train':
        st.info('The model was **trained on this customer**, so the score here looks better than it would for '
                'a brand-new customer. Pick a test-split customer to see honest behaviour.')
    else:
        st.success('This customer was **held out** of training (test split).')

    outcome = 'did not place another order' if row['observed_churn'] == 1 else 'placed another order'
    st.write(f'**Observed in the data:** in the 90 days after the cutoff this customer {outcome}.')

    st.markdown('#### Customer profile (what the model saw, before the cutoff)')
    items = list(readable_profile(row).items())
    for start in range(0, len(items), 5):
        cols = st.columns(5)
        for col, (name, value) in zip(cols, items[start:start + 5]):
            col.metric(name, value)
    st.caption('The 90-day windows are measured backwards from the cutoff date.')

    feats = manifest['features']
    with st.expander(f'All {len(feats)} model inputs (exact values scored)'):
        st.dataframe(pd.DataFrame({'feature': feats, 'value': [float(row[f]) for f in feats]}),
                     hide_index=True)


MODES = ['Type a Customer ID', 'Random customer', 'Random flagged customer']

with st.form('lookup'):
    mode = st.selectbox('How to pick a customer', MODES)
    id_text = st.text_input('Customer ID', placeholder='e.g. 12347',
                            help='Used only when "Type a Customer ID" is selected.')
    submitted = st.form_submit_button('Score customer')

if submitted:
    if mode == MODES[0]:
        row, err = find_customer(table, id_text)
    else:
        row = pick_random(table, flagged_only=(mode == MODES[2]), threshold=THRESHOLD)
        err = None if row is not None else 'Koi customer match nahi hua.'
    if err:
        st.error(err)
        st.stop()
    show_customer(row)
else:
    st.info('Pick a customer above and press **Score customer**.')