"""docs/img ki PNGs ko max 1400 px chauda karke compress karo.   python scripts/optimize_images.py"""
from pathlib import Path

from PIL import Image

for p in sorted(Path('docs/img').glob('*.png')):
    im = Image.open(p)
    if im.width > 1400:
        im = im.resize((1400, round(im.height * 1400 / im.width)), Image.LANCZOS)
    im.save(p, optimize=True)
    print(f'{p.name:32} {im.size}  {p.stat().st_size / 1024:.0f} KB')