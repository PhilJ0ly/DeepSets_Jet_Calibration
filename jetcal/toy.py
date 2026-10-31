# Philippe Joly, ATLAS Level-0 trigger ML
"""Synthetic jets with a known detector response, for tests and demos.

Truth jets from a falling spectrum fragment into constituents whose response
depends on constituent pT (threshold losses), jet eta and random smearing.
The jet response therefore depends on the fragmentation pattern: information
a set-based model can use and a pT-binned median cannot. Columns carry the
same labels as :mod:`jetcal.root_processing`, so the example configs run
unchanged on toy data.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from jetcal.data import (GLOB_LABELS, PHI_LABELS, Y_LABELS, apply_norm,
                         fit_norm, shuffle_split, validate_dataset)


def generate_toy_jets(n_jets: int, max_const: int = 25, seed: int = 0,
                      pt_range: Tuple[float, float] = (15.0, 600.0),
                      spectral_index: float = 2.5) -> dict:
    """Un-normalized toy sample following the dataset schema (no ``norm``)."""
    if n_jets < 1 or max_const < 3:
        raise ValueError("need n_jets >= 1 and max_const >= 3")
    rng = np.random.default_rng(seed)
    lo, hi = pt_range
    a = 1.0 - spectral_index  # power-law spectrum dN/dpT ~ pT^-index
    pt_true = (lo ** a + rng.uniform(size=n_jets) * (hi ** a - lo ** a)) ** (1 / a)
    eta_jet = rng.uniform(-2.8, 2.8, n_jets)
    phi_jet = rng.uniform(-np.pi, np.pi, n_jets)

    # fragmentation: multiplicity grows with log pT, Gamma-shared momentum
    n_const = np.clip(rng.poisson(4.0 + 4.0 * np.log(pt_true / lo)) + 3, 3, max_const)
    mask = (np.arange(max_const)[None] < n_const[:, None]).astype(np.float32)
    z = rng.gamma(0.8, size=(n_jets, max_const)) * mask
    z /= z.sum(1, keepdims=True)
    c_true = z * pt_true[:, None]
    c_deta = rng.normal(0, 0.15, (n_jets, max_const)) * mask
    c_dphi = rng.normal(0, 0.15, (n_jets, max_const)) * mask

    # detector response: eta non-uniformity x soft-constituent losses x smearing
    resp = ((0.62 + 0.08 * np.cos(1.3 * eta_jet))[:, None]
            * (1.0 - 0.45 * np.exp(-c_true / 3.0))
            * (1.0 + rng.normal(0, 0.08, c_true.shape)))
    c_pt = np.maximum(c_true * resp, 0.0) * mask
    c_pt[(c_pt < 0.8) & (rng.uniform(size=c_pt.shape) < 0.5)] = 0.0
    keep = np.argmax(c_true, 1)  # the hardest constituent always survives
    rows = np.arange(n_jets)
    c_pt[rows, keep] = np.maximum(c_pt[rows, keep], 0.3 * c_true[rows, keep])
    m = mask * (c_pt > 0)

    c_eta, c_phi = eta_jet[:, None] + c_deta, phi_jet[:, None] + c_dphi
    px, py = (c_pt * np.cos(c_phi)).sum(1), (c_pt * np.sin(c_phi)).sum(1)
    pz, e = (c_pt * np.sinh(c_eta)).sum(1), (c_pt * np.cosh(c_eta)).sum(1)
    pt_reco = np.hypot(px, py)
    eta_reco = np.arcsinh(pz / pt_reco)
    mass = np.sqrt(np.maximum(e ** 2 - px ** 2 - py ** 2 - pz ** 2, 0.0))
    phi_reco = np.arctan2(py, px)

    log_pt_c = np.log(np.maximum(c_pt, 1e-3))
    dphi = (c_phi - phi_reco[:, None] + np.pi) % (2 * np.pi) - np.pi
    cols = [c_pt, c_eta, c_phi, log_pt_c,
            np.log(np.maximum(c_pt / pt_reco[:, None], 1e-6)),
            c_eta - eta_reco[:, None], np.sin(dphi), np.cos(dphi)]
    x_phi = np.stack(cols, -1).astype(np.float32) * m[..., None]
    x_glob = np.stack([np.log(pt_reco), eta_reco, np.abs(eta_reco),
                       np.log1p(mass / pt_reco)], -1).astype(np.float32)
    return {"x_phi": x_phi, "mask": m.astype(np.float32), "x_glob": x_glob,
            "y": np.log(pt_true / pt_reco)[:, None].astype(np.float32),
            "pt_reco": pt_reco.astype(np.float32),
            "eta_reco": eta_reco.astype(np.float32),
            "pt_true": pt_true.astype(np.float32),
            "eta_true": eta_jet.astype(np.float32),
            "phi_lbls": list(PHI_LABELS), "glob_lbls": list(GLOB_LABELS),
            "y_lbls": list(Y_LABELS)}


def make_toy_splits(n_jets: int, max_const: int = 25, seed: int = 0,
                    test_frac: float = 0.15) -> Tuple[dict, dict]:
    """Normalized (train, test) toy splits, prepared exactly like real data."""
    train, test = shuffle_split(generate_toy_jets(n_jets, max_const, seed),
                                test_frac, seed)
    norm = {"phi": fit_norm(train["x_phi"], train["mask"]),
            "glob": fit_norm(train["x_glob"]), "y": fit_norm(train["y"])}
    for split, d in (("train", train), ("test", test)):
        d["x_phi"] = apply_norm(d["x_phi"], norm["phi"], d["mask"])
        d["x_glob"] = apply_norm(d["x_glob"], norm["glob"])
        d["norm"], d["meta"] = norm, str(dict(sample="toy", split=split, seed=seed))
        validate_dataset(d)
    return train, test
