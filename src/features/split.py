"""Ek hi jagah par train/holdout split (Day 94). Pehle notebooks aur export_artifacts mein alag-alag tha."""
import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_SEED = 42
TEST_SIZE = 0.2


def make_split(feat: pd.DataFrame, seed: int = SPLIT_SEED, test_size: float = TEST_SIZE):
    """(train_index, test_index). Customer-level (ek row = ek customer), Churned par stratified."""
    return train_test_split(feat.index, test_size=test_size, random_state=seed, stratify=feat['Churned'])