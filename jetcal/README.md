# `jetcal` — library

Library code imported by the scripts. It has no command-line entry points.
Everything passes around one **dataset dict** (schema in `data.py`) and one
**run directory** (written by `training.py`).

| module | what it does |
|---|---|
| `config.py` | `RunConfig` (model, training, features, quant), a typed JSON round-trip config. Defaults are the report's settings. |
| `data.py` | Dataset schema and validation, train-only normalization, seeded splits, feature selection by label, `.npz` I/O. |
| `root_processing.py` | ROOT ntuples to datasets (uproot/awkward): selection, cluster and jet features, book-keeping, selection QA figure. |
| `toy.py` | Synthetic jets with a known detector response, using the same labels as real data (tests and demos). |
| `models.py` | `PFNRegressor` (EnergyFlow PFN: fit, predict, differentiable `forward`, save), `load_regressor`, `BinnedMedianCorrection` baseline. |
| `training.py` | `run_training`: config + training file to a self-contained run directory. |
| `evaluation.py` | Rebuild a run's predictions as deployed (`predict_run`, `load_run_context`, `calibrated_pt`) and tabulate metrics. |
| `metrics.py` | Resolution `IQR/(2 median)`, response metrics, equal-population binning, turn-on curves and pT50/95/99. |
| `plots.py` | Performance figures: resolution vs pT, calibration maps, response spectra, eta dependence, turn-on. |
| `quantization.py` | Fixed-point emulation, quantized PFN graph, PTQ bit scans and their figures. |
| `importance.py` | Expected Gradients, permutation, occlusion, retrained ablation, and their figures. |
| `style.py` | Shared figure conventions (ATLAS style, reference-line styles, LaTeX labels, SVG + PNG saving). |

Using it from Python:

```python
from jetcal import RunConfig, load_dataset
from jetcal.training import run_training
from jetcal.evaluation import predict_run
from jetcal.plots import plot_resolution

run = run_training(RunConfig.load("configs/trigger_tiny.json"),
                   load_dataset("data/toy/toy_train.npz"), out_root="runs")
preds, d = predict_run(run, load_dataset("data/toy/toy_test.npz"))
plot_resolution(preds, d["pt_true"], save_path="resolution")
```
