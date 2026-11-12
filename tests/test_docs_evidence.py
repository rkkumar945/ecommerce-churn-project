import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH_RE = re.compile(r'`([\w./\-]+\.(?:py|md|csv|json|ipynb|txt|toml|yml))(?:::\w+)?`')
DOCS = [('docs/interview_prep.md', 15), ('docs/final_review.md', 8), ('docs/model_risk_note.md', 5)]


def _exists_exact(rel: str) -> bool:
    p = ROOT
    for part in Path(rel).parts:
        if part not in {c.name for c in p.iterdir()}:
            return False
        p = p / part
    return True


@pytest.mark.parametrize('rel,min_paths', DOCS)
def test_doc_is_complete_and_evidence_exists(rel, min_paths):
    text = (ROOT / rel).read_text(encoding='utf-8')
    assert '<<FILL' not in text, 'placeholders bache hain'
    assert text.count('```') % 2 == 0
    cited = set(PATH_RE.findall(text))
    assert len(cited) >= min_paths, f'{rel}: kam se kam {min_paths} evidence paths do'
    missing = sorted(p for p in cited if not _exists_exact(p))
    assert not missing, f'{rel}: ye paths repo mein nahi hain (ya case alag hai): {missing}'