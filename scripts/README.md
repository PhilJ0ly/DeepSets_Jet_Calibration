# `scripts` — command-line tools

One script per pipeline step, listed in the order you run them. Each one
prints its full usage with `--help`.

| script | step | inputs → outputs |
|---|---|---|
| `process_data.py` | build datasets | `.root` files (or `--toy N`) → `<calib>_{train,test}.npz`, selection QA figure |
| `train.py` | train one model | run config + train `.npz` → run directory |
| `launch_batch.sh` | train many | directory of configs → one background run per config, pinned to GPU `i mod N` |
| `evaluate.py` | evaluate | run directories + `.npz` → metrics, turn-on table and figures vs the two references |
| `dynamic_range.py` | PTQ step 1 | float run → integer bits for weights, activations and inputs, plus the scan command |
| `quant_scan.py` | PTQ step 2 | float run → bit-width scans, per-feature allocation and fidelity figures |
| `make_ptq.py` | PTQ step 3 | float run + chosen format → quantized run directory (no retraining) |
| `feature_importance.py` | input relevance | run → attribution / permutation / occlusion figures; ablation configs and collection |

Typical sequence:

```bash
python scripts/process_data.py --raw-dir $RAW --out-dir $DATA
python scripts/train.py --config configs/trigger_tiny.json --dataset $TRAIN --out-dir runs
python scripts/dynamic_range.py --run runs/trigger_tiny --dataset $TRAIN --split val
python scripts/quant_scan.py --run runs/trigger_tiny --dataset $TRAIN --split val \
    --weight-int-bits W --act-int-bits A --input-int-bits I
python scripts/make_ptq.py --run runs/trigger_tiny --weight-bits 16 --weight-int-bits W \
    --act-bits 16 --act-int-bits A --input-bits 12 --input-int-bits I
python scripts/evaluate.py --runs runs/trigger_tiny runs/trigger_tiny_ptq* --dataset $TEST
python scripts/feature_importance.py --run runs/trigger_tiny --dataset $TEST
```

Notes:
- `--split val` re-carves each run's own validation jets from the **train**
  file. Use it for every choice made during development, and keep `--split test`
  for the final numbers.
- `evaluate.py --labels` sets the legend names. Otherwise the run directory
  names are used.
- `launch_batch.sh` uses each config's `name` field as its run directory name.
