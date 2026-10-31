# `configs` — example configurations

| file | what it is |
|---|---|
| `offline_reference.json` | Offline model of the report: Φ and F of 5 × 256, `OfflineRef` inputs (jet-relative clusters, not deployable in the trigger). |
| `trigger_tiny.json` | "Tiny" trigger model: Φ 2 × 7, F 3 × 7 (281 parameters), `Trig` inputs. |
| `importance/importance.json` | Feature-importance settings: Expected Gradients, permutation within 10 reco-pT strata, occlusion. |
| `importance/groups_trigger.json` | Correlated `Trig` inputs permuted together. |

## Run config fields

Omitted fields take the defaults in `jetcal/config.py`, which are the report's
settings. Unknown keys raise an error, so typos are caught.

```jsonc
{
  "name": "trigger_tiny",            // run directory name and default legend label
  "seed": 0,                          // weights init and validation carving
  "model": {
    "phi_sizes": [7, 7],              // per-cluster Φ layers (last = latent width)
    "f_sizes": [7, 7, 7],             // post-pooling F layers
    "phi_act": "relu", "f_act": "relu"
  },
  "training": {                       // Huber loss, AdamW, early stopping
    "loss": "huber", "optimizer": "adamw", "learning_rate": 0.001,
    "epochs": 300, "batch_size": 1024, "es_patience": 40, "val_frac": 0.15
  },
  "features": {                       // labels from scripts/process_data.py
    "phi": ["log_pt_c", "eta_c"],
    "glob": ["log_pt_reco", "abs_eta", "eta", "log_m_over_pt"],
    "target": ["log_pt_over_reco"]
  }
}
```

The input sets of the report:

| name | `phi` | `glob` |
|---|---|---|
| Global4 | (the cluster inputs being scanned) | `log_pt_reco`, `abs_eta`, `eta`, `log_m_over_pt` |
| Log2 | `log_pt_c`, `eta_c` | `[]` |
| OfflineRef | `log_pt_c`, `deta_c`, `cos_dphi_c`, `sin_dphi_c`, `log_z_c` | Global4 |
| Trig | `log_pt_c`, `eta_c` | Global4 |

The PFN needs at least one cluster feature. For a scan, copy a config into a
new directory, change `name` and the field being scanned, and train the
directory with `scripts/launch_batch.sh`. Quantized runs are not configured
here: make them from a trained float run with `scripts/make_ptq.py`.
