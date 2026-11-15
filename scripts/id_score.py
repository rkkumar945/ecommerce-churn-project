import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.app import service

model, man = service.load_artifacts()
t = service.add_scores(model, service.load_table(), man['features'])
print(t.sample(3, random_state=1)[['Customer ID', 'split', 'score', 'top_share']].round(3))