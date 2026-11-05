"""Repo ke important folders. Notebook kahin se bhi chale, paths sahi rahenge."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
MODELS = ROOT / 'models'
REPORTS = ROOT / 'reports'
ARTIFACTS = ROOT / 'app_artifacts'