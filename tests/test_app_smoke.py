import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit.testing.v1 import AppTest

from src.app.service import load_table

APP = str(Path(__file__).resolve().parents[1] / "src" / "app" / "app.py")


def test_app_starts_without_exception():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert len(at.exception) == 0


def test_app_renders_a_selected_customer():
    cid = int(load_table()["Customer ID"].iloc[0])
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["selected_id"] = cid
    at.run()
    # check kya error aaya
    if len(at.error) > 0:
        print("STREAMLIT APP ERROR:", at.error[0].value)
    assert len(at.exception) == 0
    assert len(at.error) == 0
    assert any(f"Customer {cid}" in s.value for s in at.subheader)
