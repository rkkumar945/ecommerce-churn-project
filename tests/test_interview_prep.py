import re
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / 'docs' / 'interview_prep.md'
ROOT = DOC.parents[1]
PATH_RE = re.compile(r'`([\w./\-]+\.(?:py|md|csv|json|ipynb|txt|toml|yml))(?:::\w+)?`')


def _exists_exact(rel: str) -> bool:
    p = ROOT
    for part in Path(rel).parts:
        if part not in {c.name for c in p.iterdir()}:
            return False
        p = p / part
    return True


def test_no_placeholders_and_balanced_fences():
    text = DOC.read_text(encoding='utf-8')
    assert '<<FILL' not in text, 'placeholders bache hain'
    assert text.count('```') % 2 == 0


def test_every_cited_path_exists_with_exact_case():
    cited = set(PATH_RE.findall(DOC.read_text(encoding='utf-8')))
    assert len(cited) >= 15, 'kam se kam 15 evidence paths do'
    missing = sorted(p for p in cited if not _exists_exact(p))
    assert not missing, f'ye paths repo mein nahi hain (ya case alag hai): {missing}'