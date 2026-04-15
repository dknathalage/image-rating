#!/usr/bin/env python3
"""Focal bench orchestration: download / score / eval / optimize / leaderboard."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bench.dataset_ava import (
    parse_ava_txt,
    compute_mos,
    mos_to_stars,
    stratified_sample,
    download_images,
)
from bench.score import score_with_cache
from bench.musiq_scorer import score_musiq_with_cache
from bench.metrics import compute_metrics
from bench.leaderboard import regenerate_leaderboard
from bench.single_model import bucket_stars, optimize_thresholds, stars_from_thresholds


BENCH_DIR   = Path(__file__).parent
DATA_DIR    = BENCH_DIR / "data"
RESULTS_DIR = BENCH_DIR / "results"
CACHE_DIR   = BENCH_DIR / ".cache"


def git_sha_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def locate_scorer_bin() -> Path:
    """Find FocalScorer binary: env var, then DerivedData Release, then Debug."""
    env = os.environ.get("FOCAL_SCORER_BIN")
    if env:
        p = Path(env)
        if p.is_file() and p.stat().st_mode & 0o111:
            return p

    home = Path.home()
    patterns = [
        "Library/Developer/Xcode/DerivedData/**/Build/Products/Release/FocalScorer",
        "Library/Developer/Xcode/DerivedData/**/Build/Products/Debug/FocalScorer",
    ]
    for pat in patterns:
        matches = [
            p for p in home.glob(pat)
            if p.is_file() and p.stat().st_mode & 0o111
        ]
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            return matches[0]

    raise FileNotFoundError(
        "FocalScorer binary not found. Set $FOCAL_SCORER_BIN or build with: "
        "xcodebuild -scheme FocalScorer -configuration Release "
        "-destination 'platform=macOS,arch=arm64' build"
    )


def load_params(path: Path) -> dict:
    return json.loads(path.read_text())


def save_params(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def version_with_sha(version: str) -> str:
    return f"{version}@{git_sha_short()}"


def _load_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.read_csv(DATA_DIR / "ava" / "labels.csv")
    bin_ = locate_scorer_bin()
    scores = score_with_cache(bin_, DATA_DIR / "ava" / "images", CACHE_DIR)
    # Task 10 runs before Task 11: FocalScorer still emits legacy columns here.
    # Override with Python pyiqa MUSIQ as authoritative source for benches.
    # Task 11 rewrites FocalScorer to emit musiqAesthetic; simplify then.
    musiq = score_musiq_with_cache(DATA_DIR / "ava" / "images", CACHE_DIR)
    scores = scores[["filename"]].merge(musiq, on="filename", how="inner")
    return scores, labels


def cmd_download(args):
    ava_txt = DATA_DIR / "ava" / "AVA.txt"
    if not ava_txt.exists():
        raise SystemExit(
            f"AVA.txt missing at {ava_txt}. Download AVA labels first: "
            "https://github.com/mtobeiyf/ava_downloader"
        )
    df = parse_ava_txt(ava_txt)
    df = compute_mos(df)
    df = mos_to_stars(df)
    sampled = stratified_sample(df, n=args.sample, seed=0)
    out = DATA_DIR / "ava" / "images"
    downloaded = download_images(sampled, out)
    labels = downloaded[["image_id", "mos", "gt_stars", "local_path"]].copy()
    labels["filename"] = labels["local_path"].apply(lambda p: Path(p).name)
    labels[["filename", "image_id", "mos", "gt_stars"]].to_csv(
        DATA_DIR / "ava" / "labels.csv", index=False
    )
    print(f"downloaded {len(downloaded)} / {len(sampled)}; labels → data/ava/labels.csv")


def cmd_score(args):
    scores, _ = _load_dataset()
    print(scores.head())
    print(f"scored {len(scores)} images (cached under {CACHE_DIR})")


def cmd_eval(args):
    scores, labels = _load_dataset()
    payload = load_params(Path(args.params))
    thresholds = tuple(payload["thresholds"])
    df = scores.merge(labels, on="filename", how="inner")
    pred = stars_from_thresholds(df["musiqAesthetic"].to_numpy(), thresholds)
    m = compute_metrics(df["gt_stars"].to_numpy(), pred)

    version = version_with_sha(payload["version"])
    out_dir = RESULTS_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps({"val": m.to_dict()}, indent=2))
    (out_dir / "params.json").write_text(json.dumps({**payload, "date": _today()}, indent=2))
    df_out = df.copy()
    df_out["pred_stars"] = pred
    df_out[["filename", "gt_stars", "pred_stars", "musiqAesthetic"]].to_parquet(
        out_dir / "scores.parquet"
    )
    print(f"Spearman={m.spearman:.3f}  MAE={m.mae:.2f}  ±1={m.off_by_one*100:.0f}%")
    print(f"results → {out_dir}")


def cmd_optimize(args):
    scores, labels = _load_dataset()
    res = optimize_thresholds(scores, labels, n_trials=args.trials, seed=0)
    payload = {
        "version": args.version,
        "date": _today(),
        "model": "musiq-ava",
        "notes": f"optuna TPE {args.trials} trials (thresholds only)",
        "thresholds": list(res.thresholds),
    }
    out = Path(args.out) if args.out else BENCH_DIR / "params.candidate.json"
    save_params(out, payload)
    version = version_with_sha(args.version)
    out_dir = RESULTS_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps({"val": res.metrics.to_dict()}, indent=2))
    (out_dir / "params.json").write_text(json.dumps(payload, indent=2))
    print(f"best Spearman={res.metrics.spearman:.3f}  MAE={res.metrics.mae:.2f}")
    print(f"candidate → {out}")


def cmd_leaderboard(args):
    regenerate_leaderboard(RESULTS_DIR, BENCH_DIR / "LEADERBOARD.md")
    print(f"leaderboard → {BENCH_DIR / 'LEADERBOARD.md'}")


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Focal bench orchestration")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="fetch AVA subset + write labels.csv")
    d.add_argument("--sample", type=int, default=500)
    d.set_defaults(func=lambda a: cmd_download(a))

    s = sub.add_parser("score", help="run FocalScorer on images (cached)")
    s.set_defaults(func=lambda a: cmd_score(a))

    e = sub.add_parser("eval", help="evaluate a params.json against labels")
    e.add_argument("--params", default=str(BENCH_DIR / "params.current.json"))
    e.set_defaults(func=lambda a: cmd_eval(a))

    o = sub.add_parser("optimize", help="Optuna TPE search for 4 bucket thresholds")
    o.add_argument("--trials", type=int, default=500)
    o.add_argument("--version", default="v0.4.0")
    o.add_argument("--out")
    o.set_defaults(func=lambda a: cmd_optimize(a))

    l = sub.add_parser("leaderboard", help="regenerate LEADERBOARD.md")
    l.set_defaults(func=lambda a: cmd_leaderboard(a))

    return ap


def main():
    ap = _build_parser()
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
