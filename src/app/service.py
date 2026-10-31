"""App ka logic (Streamlit se alag, taaki test ho sake aur agent/API bhi reuse kar sake)."""
import json
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[2]
CARD_PATH = ROOT / 'reports' / 'final_model_card.json'
MODEL_PATH = ROOT / 'models' / 'final_model.joblib'


def load_artifacts():
    """(model, card) return karta hai, aur feature order verify karta hai."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'Model file nahi mili: {MODEL_PATH}')
    card = json.loads(CARD_PATH.read_text())
    model = joblib.load(MODEL_PATH)

    expected = getattr(model, 'feature_names_in_', None)
    if expected is not None and list(expected) != list(card['features']):
        raise ValueError('Model ke features aur model card ke features alag hain')
    return model, card