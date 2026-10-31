# Philippe Joly, ATLAS Level-0 trigger ML
"""Calibration and trigger figures of merit (no plotting).

Conventions
    * Response ``R = pT_calib / pT_true``.
    * Resolution ``IQR / (2 * median) = (q84 - q16) / (2 * q50)`` of R.
    * Binned statistics use equal-population bins; bins with fewer than
      :data:`MIN_PER_BIN` jets are blanked (NaN).
    * Turn-on: efficiency vs truth pT for ``pT_calib > threshold``;
      sharpness is ``pT99 - pT50``.
"""
from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional, Tuple, Union

import numpy as np
from scipy.stats import binned_statistic

MIN_PER_BIN = 100      # jets needed for a binned point to be drawn
MIN_TURNON_BIN = 25    # jets needed for a turn-on efficiency point


# ---------------------------------------------------------------- response

def iqr_over_2med(r: np.ndarray) -> float:
    """Relative resolution (q84 - q16) / (2 q50); sigma/mu for a Gaussian."""
    q16, med, q84 = np.percentile(r, [16, 50, 84])
    return float((q84 - q16) / (2.0 * med)) if med != 0 else float("nan")


def response_metrics(pt_pred: np.ndarray, pt_true: np.ndarray) -> Dict[str, float]:
    """Scalar summary of one calibration: median, resolution, log errors."""
    pt_pred = np.asarray(pt_pred, np.float64)
    pt_true = np.asarray(pt_true, np.float64)
    if pt_pred.shape != pt_true.shape or pt_pred.size == 0:
        raise ValueError(f"shape mismatch or empty: {pt_pred.shape} vs {pt_true.shape}")
    r = pt_pred / pt_true
    log_r = np.log(np.maximum(r, 1e-12))
    return {"median_response": float(np.median(r)),
            "resolution_iqr": iqr_over_2med(r),
            "log_response_mae": float(np.mean(np.abs(log_r))),
            "log_response_rmse": float(np.sqrt(np.mean(log_r ** 2))),
            "n_jets": int(r.size)}


# ----------------------------------------------------------------- binning

def equal_count_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-population edges over the [0.1%, 99.9%] quantiles of ``x``."""
    edges = np.unique(np.quantile(np.asarray(x, np.float64),
                                  np.linspace(0.001, 0.999, n_bins + 1)))
    if len(edges) < 3:
        raise ValueError("cannot build equal-count bins: too few distinct values")
    return edges


def log_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Log-spaced edges over the [0.1%, 99.9%] quantiles of ``x`` (> 0)."""
    lo, hi = np.quantile(np.log(np.asarray(x, np.float64)), [0.001, 0.999])
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError("cannot build log bins: degenerate range")
    return np.exp(np.linspace(lo, hi, n_bins + 1))


def binned(x: np.ndarray, values: np.ndarray, edges: np.ndarray,
           statistic: Union[str, Callable]) -> np.ndarray:
    """Per-bin statistic of ``values`` in bins of ``x``; sparse bins -> NaN."""
    stat = binned_statistic(x, values, bins=edges, statistic=statistic).statistic
    counts = np.histogram(x, bins=edges)[0]
    return np.where(counts >= MIN_PER_BIN, stat, np.nan)


def binned_quantile(x: np.ndarray, values: np.ndarray, edges: np.ndarray,
                    q: float) -> np.ndarray:
    """Per-bin quantile ``q`` (see :func:`binned`)."""
    return binned(x, values, edges, lambda v: np.quantile(v, q))


# ----------------------------------------------------------------- turn-on

def rate_matched_thresholds(predictions: Mapping[str, np.ndarray],
                            rate_fraction: float) -> Dict[str, float]:
    """Per-series threshold accepting the same fraction of all jets."""
    if not 0.0 < rate_fraction < 1.0:
        raise ValueError(f"rate_fraction must be in (0, 1), got {rate_fraction}")
    return {lab: float(np.quantile(np.asarray(p, np.float64), 1 - rate_fraction))
            for lab, p in predictions.items()}


def _crossing(centers: np.ndarray, eff: np.ndarray, level: float) -> float:
    """Truth pT where the monotonized efficiency first reaches ``level``."""
    e = np.maximum.accumulate(np.nan_to_num(eff, nan=0.0))
    i = int(np.searchsorted(e, level))
    if i == 0:
        return float(centers[0])
    if i >= len(e):
        return float("nan")
    x0, x1, y0, y1 = centers[i - 1], centers[i], e[i - 1], e[i]
    return float(x0 + (level - y0) * (x1 - x0) / max(y1 - y0, 1e-12))


def turnon_curves(predictions: Mapping[str, np.ndarray], pt_true: np.ndarray,
                  threshold_pt: Optional[float] = None,
                  rate_fraction: float = 0.05, n_bins: int = 30
                  ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray],
                             Dict[str, float], Dict[str, float]]:
    """Efficiency vs truth pT for every series.

    Working point:
        * ``threshold_pt=T`` -- every series is cut at ``T`` GeV on its own
          output (what a trigger menu does); accept fractions then differ.
        * ``threshold_pt=None`` -- every series accepts ``rate_fraction`` of
          all jets at its own threshold (equal output rate).

    Bins are fixed-width in ``[0.5, 4] x T`` (``[0.5, 2.5] x`` the median
    threshold at equal rate), coarsened automatically at low statistics.

    Returns:
        ``(edges, centers, {series: eff}, {series: threshold},
        {series: accept fraction})``.
    """
    pt_true = np.asarray(pt_true, np.float64)
    if threshold_pt is None:
        thresholds = rate_matched_thresholds(predictions, rate_fraction)
        thr = float(np.median(list(thresholds.values())))
        lo, hi = 0.5 * thr, 2.5 * thr
    else:
        thresholds = {lab: float(threshold_pt) for lab in predictions}
        lo, hi = 0.5 * threshold_pt, 4.0 * threshold_pt
    rates = {lab: float(np.mean(np.asarray(p, np.float64) > thresholds[lab]))
             for lab, p in predictions.items()}

    n_win = int(((pt_true > lo) & (pt_true < hi)).sum())
    if n_win < 4 * MIN_TURNON_BIN:
        raise ValueError(f"only {n_win} jets in the turn-on window "
                         f"[{lo:.0f}, {hi:.0f}] GeV")
    n_bins = min(n_bins, max(4, n_win // MIN_TURNON_BIN))
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    inside = (pt_true >= lo) & (pt_true < hi)
    idx = np.digitize(pt_true[inside], edges) - 1
    n = np.bincount(idx, minlength=n_bins).astype(float)

    curves = {}
    for lab, p in predictions.items():
        fired = (np.asarray(p, np.float64) > thresholds[lab])[inside]
        k = np.bincount(idx, weights=fired, minlength=n_bins)
        eff = np.divide(k, n, out=np.full_like(k, np.nan), where=n > 0)
        eff[n < MIN_TURNON_BIN] = np.nan
        curves[lab] = eff
    return edges, centers, curves, thresholds, rates


def summarize_turnon(centers: np.ndarray, eff: np.ndarray, threshold: float,
                     accept_fraction: float) -> Dict[str, float]:
    """pT50 / pT95 / pT99 crossings and sharpness (pT99 - pT50) of one curve."""
    pts = {f"pt{int(100 * lvl)}": _crossing(centers, eff, lvl)
           for lvl in (0.50, 0.95, 0.99)}
    return dict(threshold=float(threshold), accept_fraction=float(accept_fraction),
                **pts, sharpness=pts["pt99"] - pts["pt50"])


def turnon_summary(predictions: Mapping[str, np.ndarray], pt_true: np.ndarray,
                   threshold_pt: Optional[float] = None,
                   rate_fraction: float = 0.05, n_bins: int = 30
                   ) -> Dict[str, Dict[str, float]]:
    """:func:`summarize_turnon` for every series (no figure)."""
    _, centers, curves, thr, rates = turnon_curves(
        predictions, pt_true, threshold_pt, rate_fraction, n_bins)
    return {lab: summarize_turnon(centers, eff, thr[lab], rates[lab])
            for lab, eff in curves.items()}
