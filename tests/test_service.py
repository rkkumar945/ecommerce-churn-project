import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from src.app.service import MODEL_PATH, load_artifacts


@pytest.mark.skipif(not MODEL_PATH.exists(), reason='model file local-only (git-ignored)')
def test_artifacts_load_and_card_is_sane():
    _model, card = load_artifacts()
    assert len(set(card['features'])) == len(card['features'])     # duplicate columns nahi
    assert 0 < card['threshold'] < 1