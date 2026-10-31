# DeepSets jet energy calibration for the ATLAS Level-0 trigger

Code for CERN summer-student report:

Joly, P. J. R. (2026). ATLAS Level-0 Trigger Jet Energy Calibration with DeepSets. CERN. [https://doi.org/10.17181/7nja9-q3z86.](https://doi.org/10.17181/7nja9-q3z86)

A permutation-invariant DeepSets network (the EnergyFlow Particle Flow
Network, PFN) regresses the jet energy correction `log(pT_true / pT_EM)` from
the jet's topo-clusters and a few jet-level quantities. It is benchmarked
against the uncalibrated EM-scale response and a binned-median correction,
then compressed with post-training fixed-point quantization (PTQ) to check
that it fits an FPGA trigger.

```
DeepSets architecture (EnergyFlow PFN)

  clusters x_i ──► Φ (per cluster) ──► Σ_i ──┐
                                             ├──► F ──► log(pT_true / pT_EM)
  jet features g ────────────────────────────┘
```

## Repository layout

```
jetcal/            Python package (library code, no command-line entry points)
scripts/           Command-line tools, one per pipeline step
configs/           Example run configs and feature-importance settings
tests/             End-to-end test on synthetic jets
requirements.txt
```

Every folder has its own README listing what each file does.

## Installation

Python 3.10 or newer.

```bash
git clone <this repository> && cd deepsets-jet-energy-calibration
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tests/test_pipeline.py        # about 2 minutes on a CPU, should end with ALL ... PASSED
```

`energyflow >= 1.4` builds its models with `tf_keras` (Keras 2), so install a
`tf_keras` with the same minor version as `tensorflow`. The scripts add the
repository root to `sys.path`, so run them from anywhere without installing
the package.

## Quick start on synthetic jets

Runs the whole pipeline without ROOT files, in a few minutes on a CPU:

```bash
python scripts/process_data.py --toy 200000 --out-dir data/toy --calib toy
python scripts/train.py --config configs/trigger_tiny.json \
    --dataset data/toy/toy_train.npz --out-dir runs --epochs 30
python scripts/evaluate.py --runs runs/trigger_tiny --dataset data/toy/toy_test.npz
```

The figures and metrics land in `runs/trigger_tiny/summary/`.

## Full workflow

Set `DATA` to the dataset directory and `TRAIN` / `TEST` to the two files
written in step 1.

### 1. Build the datasets

```bash
python scripts/process_data.py --raw-dir $RAW_DATA_DIR --out-dir $DATA \
    --calib AntiKt4EMTopo422NoPtCutJets --pt-cut 10 --test-frac 0.15 --seed 0
TRAIN=$DATA/AntiKt4EMTopo422NoPtCutJets_train.npz
TEST=$DATA/AntiKt4EMTopo422NoPtCutJets_test.npz
```

This reads every `.root` file under `$RAW_DATA_DIR`. It keeps jets with a
matched truth jet above `--pt-cut`, builds the cluster and jet features, and
holds out a test split that is only used for the final numbers. Inputs are
standardized with statistics from the training split only. It also prints the
selection book-keeping and writes the selection QA figure (Fig. 2.1).

Available inputs (select them by label in the run config):

| block | labels |
|---|---|
| clusters (`phi`) | `pt_c`, `eta_c`, `phi_c`, `log_pt_c`, `log_z_c`, `deta_c`, `sin_dphi_c`, `cos_dphi_c` |
| jet (`glob`) | `log_pt_reco`, `eta`, `abs_eta`, `log_m_over_pt` |
| target | `log_pt_over_reco` |

`log_z_c`, `deta_c`, `sin_dphi_c` and `cos_dphi_c` are relative to the jet
axis. The jet does not exist yet when the trigger forms its cluster inputs, so
these are offline-only. Trigger models use absolute cluster quantities (the
`Trig` set: `log_pt_c`, `eta_c`).

### 2. Train

```bash
python scripts/train.py --config configs/offline_reference.json --dataset $TRAIN --out-dir runs
python scripts/train.py --config configs/trigger_tiny.json      --dataset $TRAIN --out-dir runs
# or a whole directory of configs in the background, one GPU per run:
scripts/launch_batch.sh configs/my_scan $TRAIN runs/my_scan 8
```

15% of the training file is held out for validation (`training.val_frac`,
seeded by the run seed). Each run directory is self-contained:

| file | content |
|---|---|
| `config.json` | the full run config |
| `model_spec.json` | architecture (and quantization) spec used to rebuild the graph |
| `model.weights.h5` | best-epoch weights |
| `baseline.json` | binned-median baseline (40 log bins of pT_EM), fitted on the same jets |
| `training_log.csv` | per-epoch losses |

### 3. Evaluate

```bash
# architecture / input scans on the validation split (Fig. 4.1)
python scripts/evaluate.py --runs runs/my_scan --dataset $TRAIN --split val
# final comparison on the test split (Fig. 2.2, 4.3, C.1, C.2 and Table 4.1)
python scripts/evaluate.py --runs runs/offline_reference runs/trigger_tiny_ptqW16i2A16i6I12i3 \
    --labels Offline Tiny --dataset $TEST --threshold 100
```

Each model is overlaid with the uncalibrated and binned-median references.
The script writes `<split>_metrics.json` (median response, IQR/(2 median) and
log MAE) and `<split>_turnon.json` (pT50 / pT95 / pT99 and sharpness of the
single-jet turn-on). It also prints both tables. Parameter counts are printed
by `train.py`.

### 4. Post-training quantization

Fix the range first, then the resolution:

```bash
# (a) measure the largest weights, activations and inputs -> integer bits
python scripts/dynamic_range.py --run runs/trigger_tiny --dataset $TRAIN --split val
# (b) scan the total width at those integer bits (Fig. 4.2, B.1, B.2);
#     dynamic_range.py prints this command with its numbers filled in
python scripts/quant_scan.py --run runs/trigger_tiny --dataset $TRAIN --split val \
    --weight-bits 8 10 12 14 16 --weight-int-bits 2 --act-int-bits 6 \
    --input-bits 6 8 10 12 14 16 --input-int-bits 3 \
    --fidelity-weight-bits 16 --fidelity-input-bits 12 --fidelity-rate 0.15
# (c) freeze the chosen format into a new run directory (no retraining)
python scripts/make_ptq.py --run runs/trigger_tiny \
    --weight-bits 16 --weight-int-bits 2 --act-bits 16 --act-int-bits 6 \
    --input-bits 12 --input-int-bits 3
```

The PTQ run directory (`runs/trigger_tiny_ptqW16i2A16i6I12i3`) evaluates like
any other run.

**Fixed-point notation.** In the code, `Q(bits, int_bits)` means 1 sign bit,
`int_bits` integer bits and `bits - 1 - int_bits` fractional bits. Values
saturate at `[-2^int_bits, 2^int_bits)`. The report quotes
*(total, integer including the sign bit)*. Report (16, 3) is therefore
`--*-bits 16 --*-int-bits 2`. Biases use the activation integer bits.

### 5. Feature importance

```bash
python scripts/feature_importance.py --run runs/trigger_tiny_ptqW16i2A16i6I12i3 --dataset $TEST \
    --config configs/importance/importance.json \
    --groups configs/importance/groups_trigger.json
# retrained leave-one-out ablation: write configs, train them, collect them
python scripts/feature_importance.py --run runs/trigger_tiny --dataset $TEST \
    --ablation-configs configs/ablation --ablation-seeds 0 1 2
scripts/launch_batch.sh configs/ablation $TRAIN runs/ablation
python scripts/feature_importance.py --run runs/trigger_tiny --dataset $TEST \
    --ablation-runs runs/ablation
```

The study uses four probes:
- **Expected Gradients:** what the network uses, per jet, in % of calibrated pT.
- **Permutation:** how much resolution depends on the feature.
- **Occlusion:** the feature held at its mean.
- **Retrained ablation:** the decisive test of whether an input can be dropped.

Permute correlated inputs together with `--groups`. The study always runs
with the run's own inputs and fixed-point settings.

## Report figures and where they come from

| report | output file | command |
|---|---|---|
| Fig. 2.1 | `$DATA/qa/<calib>_selection_qa` | `process_data.py` |
| Fig. 2.2 | `<split>_uncalibrated_calibration` | `evaluate.py` |
| Fig. 4.1 | `val_resolution` (or `val_resolution_vs_reco`) | `evaluate.py --split val` |
| Fig. 4.2 | `val_quant_weight_scan`, `val_quant_input_scan` | `quant_scan.py` |
| Fig. 4.3 | `test_resolution`, `test_turnon_100GeV` | `evaluate.py` |
| Table 4.1 | `test_metrics.json`, `test_turnon.json` | `evaluate.py` |
| Fig. B.1 | `val_quant_fidelity` | `quant_scan.py --fidelity-rate 0.15` |
| Fig. B.2 | `val_quant_bit_allocation` | `quant_scan.py` |
| Fig. C.1 | `test_response_spectra` | `evaluate.py` |
| Fig. C.2 | `test_eta_dependence` | `evaluate.py` |

Every figure is written as `.svg` and as a 400 dpi `.png`.

## Conventions

- **Response:** `R = pT_calib / pT_true`.
- **Resolution:** `IQR / (2 × median) = (q84 − q16) / (2 q50)` of R.
- **Closure:** median R.
- **Binning:** binned statistics use equal-population bins. Bins with fewer
  than 100 jets are not drawn.
- **Turn-on:** every series is cut at the same threshold in GeV on its own
  calibrated pT, as a trigger menu would apply it. Sharpness is `pT99 − pT50`.
- **Comparing runs:** two configurations differ only if a metric differs by
  more than 2σ_seed, the spread over training seeds. Train the same config
  with `--seed 0/1/2` to measure it.

## AI Disclaimer

The `README.md` files and comments were gernerated using [Claude Opus 5.5](https://claude.ai/). Claude was also used to generate the plotting functions, the testing suite, and the `BASH` scripts.

## Author

Philippe Joly (McGill / UBC / TRIUMF / CERN), supervised by C. Gay,
M. Swiatlowski and K. Leong.
