#!/usr/bin/env python3
# Philippe Joly, ATLAS Level-0 trigger ML
"""Turn a trained float run into a post-training-quantized run (no retraining).

PTQ only changes the graph that consumes the weights, so the PTQ run is the
float run's files plus a quantization block in model_spec.json and
config.json. evaluate.py and feature_importance.py then treat it like any
other run.

Output: <run>_ptqW<b>i<i>A<b>i<i>I<b>i<i>/ (or --out)

Example:
    python scripts/make_ptq.py --run runs/trigger_tiny \\
        --weight-bits 16 --weight-int-bits 2 --act-bits 16 --act-int-bits 6 \\
        --input-bits 12 --input-int-bits 3
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jetcal.config import QuantConfig  # noqa: E402

_COPY = ("model.weights.h5", "baseline.json", "training_log.csv")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="trained float run directory")
    p.add_argument("--out", help="output run directory")
    for name in ("weight", "act", "input"):
        p.add_argument(f"--{name}-bits", type=int, required=True)
        p.add_argument(f"--{name}-int-bits", type=int, required=True)
    p.add_argument("--components", nargs="+", default=["phi", "f", "output"])
    a = p.parse_args()

    run = Path(a.run)
    spec = json.loads((run / "model_spec.json").read_text())
    cfg = json.loads((run / "config.json").read_text())
    if (spec.get("quant") or {}).get("mode", "off") != "off":
        p.error(f"{run.name} is already quantized; start from the float run")
    for name in ("weight", "act", "input"):
        bits, ib = getattr(a, f"{name}_bits"), getattr(a, f"{name}_int_bits")
        if bits - 1 - ib < 1:
            p.error(f"{name}: Q({bits}, {ib}) leaves no fractional bit")
    q = QuantConfig(mode="ptq", components=tuple(a.components),
                    weight_bits=a.weight_bits, weight_int_bits=a.weight_int_bits,
                    act_bits=a.act_bits, act_int_bits=a.act_int_bits,
                    input_phi_bits=a.input_bits, input_glob_bits=a.input_bits,
                    input_int_bits=a.input_int_bits)
    quant = {**q.__dict__, "components": list(q.components)}

    tag = (f"ptqW{a.weight_bits}i{a.weight_int_bits}A{a.act_bits}i{a.act_int_bits}"
           f"I{a.input_bits}i{a.input_int_bits}")
    out = Path(a.out) if a.out else run.with_name(f"{run.name}_{tag}")
    out.mkdir(parents=True, exist_ok=True)
    for name in _COPY:
        if (run / name).is_file():
            shutil.copy2(run / name, out / name)
    (out / "model_spec.json").write_text(json.dumps({**spec, "quant": quant}, indent=2) + "\n")
    cfg.update(name=out.name, quant=quant)
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"[ptq] {run.name} -> {out}\n  {q.describe()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
