#!/usr/bin/env python3
# Philippe Joly, ATLAS Level-0 trigger ML
"""Which inputs matter: attribution, permutation, occlusion, retrained ablation.

Modes (combinable):
    (default)            Expected Gradients + permutation + occlusion on one run
    --ablation-configs   write leave-one-out run configs, then stop
    --ablation-runs DIR  add the trained abl_* runs in DIR to the study

Outputs (in --out, default <run>/importance):
    <split>_importance.json, <split>_importance_ranking, <split>_importance_beeswarm,
    <split>_importance_vs_pt, <split>_importance_ablation (with --ablation-runs)

Example:
    python scripts/feature_importance.py --run runs/trigger_tiny --dataset $TEST \\
        --config configs/importance/importance.json \\
        --groups configs/importance/groups_trigger.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jetcal.data import load_dataset  # noqa: E402
from jetcal.importance import (ImportanceConfig, build_ablation_configs,  # noqa: E402
                               collect_ablation, plot_ablation,
                               plot_attribution_beeswarm, plot_importance_ranking,
                               plot_importance_vs_pt, run_importance, write_records)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="trained (or PTQ) run directory")
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="test", choices=("val", "test"))
    p.add_argument("--config", help="ImportanceConfig JSON (defaults otherwise)")
    p.add_argument("--groups", help="JSON {group: [\"const:x\", \"glob:y\"]} permuted together")
    p.add_argument("--max-jets", type=int, help="override max_jets")
    p.add_argument("--out")
    p.add_argument("--ablation-configs", metavar="DIR", help="write ablation configs to DIR")
    p.add_argument("--ablation-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--ablation-runs", metavar="DIR", help="trained abl_* runs to collect")
    a = p.parse_args()

    run = Path(a.run)
    if a.ablation_configs:
        build_ablation_configs(run, a.ablation_configs, seeds=a.ablation_seeds)
        return 0
    cfg = json.loads(Path(a.config).read_text()) if a.config else {}
    if a.groups:
        cfg["groups"] = json.loads(Path(a.groups).read_text())
    if a.max_jets:
        cfg["max_jets"] = a.max_jets
    cfg = ImportanceConfig.from_dict(cfg)

    out = Path(a.out) if a.out else run / "importance"
    out.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(a.dataset)
    rec, arrays = run_importance(run, dataset, a.split, cfg)
    if a.ablation_runs:
        rec["ablation"] = collect_ablation(a.ablation_runs, dataset, a.split, cfg.fom)
    write_records(rec, out / f"{a.split}_importance.json")

    s = a.split
    plot_importance_ranking(rec, out / f"{s}_importance_ranking")
    plot_importance_vs_pt(rec, save_path=out / f"{s}_importance_vs_pt")
    if arrays:
        plot_attribution_beeswarm(rec, arrays, save_path=out / f"{s}_importance_beeswarm")
    if "ablation" in rec:
        plot_ablation(rec["ablation"], cfg.fom[0], out / f"{s}_importance_ablation")
    print(f"[importance] done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
