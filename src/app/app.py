import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:          # streamlit script ki apni folder ko path banata hai, repo root ko nahi
    sys.path.insert(0, str(ROOT))

from src.app.service import load_artifacts

st.set_page_config(page_title='Churn Risk Explorer', page_icon='📉', layout='wide')


@st.cache_resource
def get_artifacts():
    return load_artifacts()


st.title('Customer Churn Risk Explorer')
st.caption('Ranks customers by their risk of not ordering in the next 90 days.')

try:
    model, card = get_artifacts()

    # Sidebar waala sara code try block ke ANDAR hona chahiye
    with st.sidebar:
        st.header('Model')
        st.write(f"**")
        st.metric('Test ROC-AUC', f"{card['test_auc']:.3f}")
        st.metric('Alert threshold', f"{card['threshold']:.2f}")
        st.caption('Probabilities are for ranking, not calibrated risk percentages.')

except Exception as e:  # noqa: BLE001
    st.error(f'Model load nahi hua: {e}')
    st.stop()


st.info('Skeleton chal raha hai. Customer lookup Day 88-89 mein aayega.')