import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / 'README.md'
CARD = ROOT / 'reports' / 'final_model_card.json'
DEPLOY = ROOT / 'docs' / 'deployment.md'
LIVE_URL = 'https://ecommerce-churn-project.streamlit.app'
MAX_IMG_KB = 600
BANNED = ['<<FILL', '[X]', 'TODO', 'FIXME', '{{', 'lorem ipsum']
IMG_EXT = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}


def _text() -> str:
    return README.read_text(encoding='utf-8')


def _placeholder_tokens(text: str) -> list:
    non_task_list_text = '\n'.join(
        line for line in text.splitlines()
        if not re.match(r'^\s*[-*+]\s+\[[ xX]\]', line)
    )
    found = [
        b for b in BANNED
        if b != '[X]' and b.lower() in non_task_list_text.lower()
    ]
    if '[X]' in text:
        found.append('[X]')
    return found


def _prose() -> str:
    """Code fences hata ke: unke andar ke '# comment' headings nahi hote aur links check nahi hote."""
    return re.sub(r'```.*?```', '', _text(), flags=re.DOTALL)


def _slug(title: str) -> str:
    return re.sub(r'[^\w\- ]', '', title.strip().lower()).replace(' ', '-')


def _anchors() -> set:
    seen, out = {}, set()
    for m in re.finditer(r'^#{1,6}\s+(.+?)\s*$', _prose(), flags=re.MULTILINE):
        slug = _slug(m.group(1))
        n = seen.get(slug, 0)
        out.add(slug if n == 0 else f'{slug}-{n}')
        seen[slug] = n + 1
    return out


def _targets() -> list:
    prose = _prose()
    return re.findall(r'!?\[[^\]]*\]\(([^)\s]+)', prose) + re.findall(r'(?:src|href)="([^"]+)"', prose)


def _exists_exact(rel: str) -> bool:
    """Case-sensitive check, jaisa GitHub (Linux) karta hai. Windows par Path.exists() case ignore karta hai."""
    p = ROOT
    for part in Path(rel).parts:
        if part not in {c.name for c in p.iterdir()}:
            return False
        p = p / part
    return True


def test_no_placeholders_and_balanced_fences():
    text = _text()
    found = _placeholder_tokens(text)
    assert not found, f'placeholders bache hain: {found}'
    assert text.count('```') % 2 == 0, 'code fence band nahi hua'


def test_task_list_checkboxes_are_not_placeholders():
    assert not _placeholder_tokens('- [x] done\n- [ ] todo')


def test_live_url_present_and_consistent():
    assert _text().count(LIVE_URL) >= 2            # badge + link
    if DEPLOY.exists():
        assert LIVE_URL in DEPLOY.read_text(encoding='utf-8')


def test_relative_links_and_images_resolve_with_exact_case():
    anchors = _anchors()
    bad = []
    for t in _targets():
        if t.startswith(('http://', 'https://', 'mailto:')):
            continue
        if t.startswith('#'):
            if t[1:] not in anchors:
                bad.append(t)
            continue
        path = t.split('#')[0]
        if path and not _exists_exact(path):
            bad.append(t)
    assert not bad, f'broken links: {bad}'


def test_images_are_light_enough():
    for t in _targets():
        p = ROOT / t.split('#')[0]
        if not t.startswith('http') and p.suffix.lower() in IMG_EXT:
            assert p.stat().st_size <= MAX_IMG_KB * 1024, f'{t}: {p.stat().st_size // 1024} KB, chhota karo'


@pytest.mark.skipif(not CARD.exists(), reason='model card missing')
def test_headline_numbers_match_the_model_card():
    card = json.loads(CARD.read_text())
    text = _text()
    assert card['model'] in text
    assert f"{card['test_auc']:.3f}" in text, 'README ka test AUC model card se match nahi karta'
    assert f"{card['threshold']:.2f}" in text


def test_required_sections_exist():
    heads = {m.group(1).strip().lower() for m in re.finditer(r'^##\s+(.+?)\s*$', _prose(), flags=re.MULTILINE)}
    for needed in ('limitations', 'how to run', 'results', 'roadmap', 'system architecture'):
        assert any(needed in h for h in heads), f'section missing: {needed}'