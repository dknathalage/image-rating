from pathlib import Path
import numpy as np
from bench.dataset_ava import parse_ava_txt, compute_mos, stratified_sample, mos_to_stars

FIXTURE = Path(__file__).parent / "fixtures" / "mini_ava.txt"

def test_parse_ava_txt():
    df = parse_ava_txt(FIXTURE)
    assert len(df) == 5
    assert "image_id" in df.columns
    assert all(f"count_{i}" in df.columns for i in range(1, 11))

def test_compute_mos():
    df = parse_ava_txt(FIXTURE)
    df = compute_mos(df)
    expected = (1*0 + 2*1 + 3*5 + 4*7 + 5*10 + 6*15 + 7*20 + 8*25 + 9*10 + 10*7) / 100
    assert abs(df.loc[df.image_id == 953619, "mos"].iloc[0] - expected) < 1e-6

def test_mos_to_stars_quintile():
    df = parse_ava_txt(FIXTURE)
    df = compute_mos(df)
    df = mos_to_stars(df)
    assert df.gt_stars.min() == 1
    assert df.gt_stars.max() == 5
    max_row = df.loc[df.mos.idxmax()]
    min_row = df.loc[df.mos.idxmin()]
    assert max_row.gt_stars == 5
    assert min_row.gt_stars == 1

def test_stratified_sample_balanced():
    df = parse_ava_txt(FIXTURE)
    df = compute_mos(df)
    df = mos_to_stars(df)
    sampled = stratified_sample(df, n=5, seed=0)
    assert len(sampled) == 5
    assert set(sampled.gt_stars.unique()) == {1, 2, 3, 4, 5}
