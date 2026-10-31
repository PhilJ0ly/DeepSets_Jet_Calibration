# Philippe Joly, ATLAS Level-0 trigger ML
"""Dataset dict: schema, normalization, splits, feature selection, I/O.

Schema (the one in-memory format used everywhere)
    x_phi   (N, C, P) float32  per-constituent features, standardized, 0 on pads
    mask    (N, C)    float32  1 = real constituent, 0 = padding
    x_glob  (N, G)    float32  jet-level features, standardized
    y       (N, T)    float32  regression target(s), NOT standardized
    pt_reco, pt_true, eta_reco, eta_true   (N,) diagnostics, never inputs
    phi_lbls, glob_lbls, y_lbls            labels of the last axes above
    norm    {"phi"|"glob"|"y": {"mean", "std"}}  fitted on the TRAIN split only
    meta    provenance string
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

#: columns written by :mod:`jetcal.root_processing` (and :mod:`jetcal.toy`)
PHI_LABELS = ["pt_c", "eta_c", "phi_c", "log_pt_c", "log_z_c",
              "deta_c", "sin_dphi_c", "cos_dphi_c"]
GLOB_LABELS = ["log_pt_reco", "eta", "abs_eta", "log_m_over_pt"]
Y_LABELS = ["log_pt_over_reco"]

_LABEL_KEYS = ("phi_lbls", "glob_lbls", "y_lbls")
_META_KEYS = set(_LABEL_KEYS) | {"meta"}


def validate_dataset(d: dict, require_norm: bool = True) -> None:
    """Raise ``KeyError``/``ValueError`` if ``d`` violates the schema."""
    for key in ("x_phi", "mask", "x_glob", "y", *_LABEL_KEYS):
        if key not in d:
            raise KeyError(f"dataset missing required key {key!r}")
    x_phi, mask, x_glob, y = d["x_phi"], d["mask"], d["x_glob"], d["y"]
    if (x_phi.ndim, mask.ndim, x_glob.ndim, y.ndim) != (3, 2, 2, 2):
        raise ValueError(f"bad ranks: x_phi{x_phi.shape} mask{mask.shape} "
                         f"x_glob{x_glob.shape} y{y.shape}")
    if not len(x_phi) == len(mask) == len(x_glob) == len(y):
        raise ValueError("row-count mismatch between x_phi, mask, x_glob, y")
    if x_phi.shape[1] != mask.shape[1]:
        raise ValueError("constituent-count mismatch between x_phi and mask")
    for key, arr, axis in (("phi_lbls", x_phi, 2), ("glob_lbls", x_glob, 1),
                           ("y_lbls", y, 1)):
        if arr.shape[axis] != len(d[key]):
            raise ValueError(f"{key}: {len(d[key])} labels for {arr.shape[axis]} columns")
    for key in ("x_phi", "mask", "x_glob", "y"):
        if not np.isfinite(d[key]).all():
            raise ValueError(f"non-finite values in {key!r}")
    if require_norm:
        for lvl in ("phi", "glob", "y"):
            stats = d.get("norm", {}).get(lvl, {})
            if {"mean", "std"} - set(stats):
                raise KeyError(f"norm[{lvl!r}] must provide 'mean' and 'std'")
            if np.any(np.asarray(stats["std"]) <= 0):
                raise ValueError(f"norm[{lvl!r}]['std'] must be > 0")


# ------------------------------------------------------------ normalization

def fit_norm(x: np.ndarray, mask: Optional[np.ndarray] = None,
             eps: float = 1e-6) -> Dict[str, np.ndarray]:
    """Per-feature mean/std; with ``mask`` only real constituents count."""
    x = np.asarray(x, np.float64)
    flat = x.reshape(-1, x.shape[-1]) if mask is None else x[np.asarray(mask, bool)]
    if flat.size == 0:
        raise ValueError("no entries to fit normalization on")
    return {"mean": flat.mean(axis=0), "std": np.maximum(flat.std(axis=0), eps)}


def apply_norm(x: np.ndarray, norm: Dict[str, np.ndarray],
               mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Standardize; re-zero padded rows when ``mask`` is given."""
    z = ((np.asarray(x) - norm["mean"]) / norm["std"]).astype(np.float32)
    return z * np.asarray(mask, np.float32)[..., None] if mask is not None else z


def standardize_targets(y: np.ndarray, norm_y: Dict[str, np.ndarray]) -> np.ndarray:
    y = np.asarray(y, np.float64).reshape(len(y), -1)
    return ((y - norm_y["mean"]) / norm_y["std"]).astype(np.float32)


def unstandardize_targets(z: np.ndarray, norm_y: Dict[str, np.ndarray]) -> np.ndarray:
    z = np.asarray(z, np.float64).reshape(len(z), -1)
    return z * norm_y["std"] + norm_y["mean"]


def invert_target(y_hat: np.ndarray, pt_reco: np.ndarray) -> np.ndarray:
    """Calibrated pT from a predicted log(pT_true / pT_reco)."""
    return np.asarray(pt_reco, np.float64) * np.exp(np.asarray(y_hat).ravel())


# ---------------------------------------------------------------- splitting

def take_rows(d: dict, idx: np.ndarray) -> dict:
    """Row-index every per-jet array of ``d``; labels/norm/meta pass through."""
    n = len(d["y"])
    return {k: (v[idx] if isinstance(v, np.ndarray) and len(v) == n else v)
            for k, v in d.items()}


def _split(d: dict, frac: float, seed: int) -> Tuple[dict, dict]:
    if not 0.0 < frac < 1.0:
        raise ValueError(f"split fraction {frac} must be in (0, 1)")
    idx = np.random.default_rng(seed).permutation(len(d["y"]))
    cut = int(round((1.0 - frac) * len(idx)))
    return take_rows(d, idx[:cut]), take_rows(d, idx[cut:])


def shuffle_split(d: dict, test_frac: float = 0.15, seed: int = 0) -> Tuple[dict, dict]:
    """Seeded (train, test) split; shuffling mixes the JZ slices."""
    return _split(d, test_frac, seed)


def carve_validation(d: dict, val_frac: float, seed: int) -> Tuple[dict, dict]:
    """Seeded (fit, val) split of the training file, reproducible per run."""
    return _split(d, val_frac, seed)


def subsample(d: dict, max_jets: Optional[int], seed: int = 0) -> dict:
    """Seeded random subset of at most ``max_jets`` jets (order preserved)."""
    n = len(d["y"])
    if max_jets is None or n <= max_jets:
        return d
    idx = np.sort(np.random.default_rng(seed).choice(n, int(max_jets), replace=False))
    return take_rows(d, idx)


# ---------------------------------------------------------- feature selection

def _indices(available: Sequence[str], wanted: Sequence[str], what: str) -> list:
    missing = [w for w in wanted if w not in available]
    if missing:
        raise KeyError(f"{what} features {missing} not in dataset; "
                       f"available: {list(available)}")
    return [list(available).index(w) for w in wanted]


def select_features(d: dict, phi: Optional[Sequence[str]] = None,
                    glob: Optional[Sequence[str]] = None,
                    target: Optional[Sequence[str]] = None,
                    verbose: bool = True) -> dict:
    """Copy of ``d`` restricted (and ordered) to the requested labels.

    ``None`` keeps every column of that block; ``norm`` is sliced to match.
    """
    validate_dataset(d, require_norm="norm" in d)
    sel = {"phi": list(d["phi_lbls"]) if phi is None else list(phi),
           "glob": list(d["glob_lbls"]) if glob is None else list(glob),
           "y": list(d["y_lbls"]) if target is None else list(target)}
    idx = {"phi": _indices(d["phi_lbls"], sel["phi"], "phi"),
           "glob": _indices(d["glob_lbls"], sel["glob"], "glob"),
           "y": _indices(d["y_lbls"], sel["y"], "target")}
    out = dict(d)
    for key, lvl in (("x_phi", "phi"), ("x_glob", "glob"), ("y", "y")):
        out[key] = np.ascontiguousarray(d[key][..., idx[lvl]])
    out["phi_lbls"], out["glob_lbls"], out["y_lbls"] = sel["phi"], sel["glob"], sel["y"]
    if "norm" in d:
        out["norm"] = {lvl: {k: np.asarray(v)[..., idx[lvl]]
                             for k, v in d["norm"][lvl].items()}
                       for lvl in ("phi", "glob", "y")}
    if verbose:
        for lvl, labels in sel.items():
            print(f"[select] {lvl:<5} ({len(labels)}): {labels}")
    return out


# ---------------------------------------------------------------------- I/O

def concat_datasets(parts: Sequence[dict]) -> dict:
    """Row-wise concatenation of per-file dicts (array keys only)."""
    if not parts:
        raise ValueError("no dataset parts to concatenate")
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def save_dataset(path: Union[str, Path], d: dict) -> Path:
    """Write one split as a self-contained compressed ``.npz``."""
    validate_dataset(d)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in d.items()
              if isinstance(v, np.ndarray) and k not in _META_KEYS}
    np.savez_compressed(
        path, **arrays, **{k: np.asarray(d[k]) for k in _LABEL_KEYS},
        meta=np.asarray(str(d.get("meta", ""))),
        **{f"norm_{lvl}_{s}": np.asarray(d["norm"][lvl][s])
           for lvl in ("phi", "glob", "y") for s in ("mean", "std")})
    return path


def load_dataset(path: Union[str, Path]) -> dict:
    """Read and validate a split written by :func:`save_dataset`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"dataset not found: {path}")
    with np.load(path, allow_pickle=False) as f:
        d = {k: f[k] for k in f.files if k not in _META_KEYS
             and not k.startswith("norm_")}
        for key in _LABEL_KEYS:
            d[key] = [str(s) for s in f[key]]
        d["norm"] = {lvl: {s: f[f"norm_{lvl}_{s}"] for s in ("mean", "std")}
                     for lvl in ("phi", "glob", "y")}
        d["meta"] = str(f["meta"])
    validate_dataset(d)
    return d
