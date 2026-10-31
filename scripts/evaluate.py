#!/usr/bin/env python3
# Philippe Joly, ATLAS Level-0 trigger ML
"""Evaluate trained runs against the uncalibrated and binned-median references.

--runs takes run directories and/or parents of run directories. Every model
is overlaid with the two references (taken from the first run).

Outputs (in --out, default <first --runs>/summary):
    <split>_metrics.json             median response, IQR/(2 median), log MAE
    <split>_turnon.json              pT50/95/99 and sharpness per series
    <split>_resolution               resolution + median vs truth pT
    <split>_resolution_vs_reco       same vs uncalibrated pT_EM
    <split>_response_spectra         response shapes in truth-pT windows
    <split>_eta_dependence           median + resolution vs eta
    <split>_turnon_<T>GeV            single-jet turn-on at --threshold
    <split>_uncalibrated_calibration 2D response maps of the uncalibrated pT
  and <run>/<split>_metrics.json per run.

Example:
    python scripts/evaluate.py --runs runs/offline_reference runs/trigger_tiny \\
        --labels Offline Tiny --dataset $DATA/AntiKt4EMTopo422NoPtCutJets_test.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jetcal.config import RunConfig  # noqa: E402
from jetcal.data import load_dataset  # noqa: E402
from jetcal.evaluation import evaluate_predictions, predict_run  # noqa: E402
from jetcal import plots  # noqa: E402


def find_runs(paths: List[str]) -> List[Path]:
    """Run directories (holding config.json) among ``paths`` or their children."""
    runs = []
    for p in map(Path, paths):
        if (p / "config.json").is_file():
            runs.append(p)
        else:
            runs += sorted(c for c in p.iterdir() if (c / "config.json").is_file())
    if not runs:
        raise FileNotFoundError(f"no run directories found in {paths}")
    return runs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", help="legend label per run (default: run name)")
    p.add_argument("--dataset", required=True,
                   help="test .npz (or the train .npz with --split val)")
    p.add_argument("--split", default="test", choices=("val", "test"),
                   help="'val' re-carves each run's validation jets from the train file")
    p.add_argument("--threshold", type=float, default=100.0,
                   help="single-jet trigger threshold [GeV]")
    p.add_argument("--out", help="summary directory")
    a = p.parse_args()

    runs = find_runs(a.runs)
    labels = a.labels or [r.name for r in runs]
    if len(labels) != len(runs):
        p.error(f"{len(labels)} labels for {len(runs)} runs: {[r.name for r in runs]}")
    if a.split == "val" and len({(RunConfig.load(r / "config.json").seed,
                                  RunConfig.load(r / "config.json").training.val_frac)
                                 for r in runs}) > 1:
        p.error("runs carve different validation splits; overlay them on --split test")

    dataset = load_dataset(a.dataset)
    combined, d = {}, None
    for run, label in zip(runs, labels):
        print(f"\n[evaluate] {run}")
        preds, d_run = predict_run(run, dataset, a.split, label=label)
        evaluate_predictions(preds, d_run["pt_true"], run, a.split)
        if d is None:
            d = d_run
            combined = {k: preds[k] for k in ("Uncalibrated", "Binned median")}
        combined[label] = preds[label]

    out = Path(a.out) if a.out else Path(a.runs[0]) / "summary"
    s, pt_true = a.split, d["pt_true"]
    evaluate_predictions(combined, pt_true, out, s)
    plots.plot_resolution(combined, pt_true, save_path=out / f"{s}_resolution")
    plots.plot_resolution_vs_reco(combined, d["pt_reco"], pt_true,
                                  save_path=out / f"{s}_resolution_vs_reco")
    plots.plot_response_spectra(combined, pt_true, save_path=out / f"{s}_response_spectra")
    plots.plot_eta_dependence(combined, pt_true, d["eta_true"],
                              save_path=out / f"{s}_eta_dependence")
    plots.plot_calibration(pt_true, combined["Uncalibrated"], d["eta_true"], label="EM",
                           save_path=out / f"{s}_uncalibrated_calibration")
    _, turnon = plots.plot_turnon(combined, pt_true, threshold_pt=a.threshold,
                                  save_path=out / f"{s}_turnon_{a.threshold:g}GeV")
    (out / f"{s}_turnon.json").write_text(json.dumps(turnon, indent=2) + "\n")
    print(f"\n[evaluate] single jet > {a.threshold:g} GeV")
    for lab, t in turnon.items():
        print(f"  {lab:<20} accept={t['accept_fraction']:7.3%}  pT50={t['pt50']:6.1f}  "
              f"pT95={t['pt95']:6.1f}  pT99={t['pt99']:6.1f}  "
              f"sharpness={t['sharpness']:5.1f} GeV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
