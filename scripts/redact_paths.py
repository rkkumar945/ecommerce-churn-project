"""Notebook source/outputs se local user paths hatao (C:\\Users\\<name> -> C:\\Users\\<user>). Cells re-run nahi hote.
    python scripts/redact_paths.py            # dry-run
    python scripts/redact_paths.py --apply
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [
    (re.compile(r'[A-Za-z]:[\\/]+Users[\\/]+(?!<user>)[^\\/\s"\'<>|:*?]+'), 'C:\\Users\\<user>'),
    (re.compile(r'/Users/(?!<user>)[^/\s"\']+'), '/Users/<user>'),
    (re.compile(r'/home/(?!<user>)[^/\s"\']+'), '/home/<user>'),
]


def scrub(obj, stats):
    if isinstance(obj, str):
        for pat, repl in PATTERNS:
            obj, n = pat.subn(lambda m, r=repl: r, obj)
            stats[0] += n
        return obj
    if isinstance(obj, list):
        return [scrub(x, stats) for x in obj]
    if isinstance(obj, dict):
        return {k: scrub(v, stats) for k, v in obj.items()}
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args()
    out = subprocess.run(['git', 'ls-files', '*.ipynb'], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    total = 0
    for rel in filter(None, out.split('\n')):
        p = ROOT / rel
        nb = json.loads(p.read_text(encoding='utf-8'))
        stats = [0]
        new = scrub(nb, stats)
        if stats[0]:
            total += stats[0]
            print(f'{rel}: {stats[0]} replacements')
            if a.apply:
                with open(p, 'w', encoding='utf-8', newline='\n') as fh:
                    fh.write(json.dumps(new, indent=1, ensure_ascii=False) + '\n')
    print(('applied' if a.apply else 'dry-run (--apply se likho)'), '| total replacements:', total)


if __name__ == '__main__':
    main()