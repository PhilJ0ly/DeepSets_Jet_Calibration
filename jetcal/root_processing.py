# Philippe Joly, ATLAS Level-0 trigger ML
"""ROOT ntuples -> normalized train/test datasets (uproot + awkward).

Per jet
    [phi]  constituent clusters: pT, eta, phi, log pT, log z = log(pT_c/pT_jet),
           and jet-relative deta, sin/cos dphi (jet-relative ones: offline only)
    [glob] log pT_EM, eta, |eta|, log(1 + m/pT)
    [y]    log(pT_true / pT_EM)

Truth branches are used for the selection and the target only, never as
inputs. Every production prints a per-file book-keeping table (unique events,
jets, matched, selected, efficiency) and writes the selection QA figure:
stacked selected/rejected histograms of truth pT, eta, phi and mu over
truth-matched jets, with the selected fraction underneath.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Dict, Iterator, Sequence, Tuple, Union

import numpy as np

try:
    import awkward as ak
    import uproot
except ImportError as _err:  # pragma: no cover
    raise ImportError("jetcal.root_processing needs 'uproot' and 'awkward'") from _err

from jetcal.data import (GLOB_LABELS, PHI_LABELS, Y_LABELS, apply_norm,
                         concat_datasets, fit_norm, save_dataset,
                         shuffle_split, validate_dataset)

_C_SEL, _C_REJ = "#1f77b4", "#d62728"  # blue = selected, red = rejected


@dataclass
class ProcessingConfig:
    """Everything that defines one dataset production."""

    calib: str = "AntiKt4EMTopo422NoPtCutJets"   # jet collection branch prefix
    tree_name: str = "ntuple"
    max_const: int = 80          # constituent padding length
    pt_cut: float = 10.0         # GeV, on the matched truth jet
    test_frac: float = 0.15
    seed: int = 0
    qa_plots: bool = True
    branches: Tuple[str, ...] = (
        "pt", "eta", "phi", "m", "nConstituents", "constituentPt",
        "constituentEta", "constituentPhi", "matchedTruth_pt", "matchedTruth_eta")
    # book-keeping only; run/event numbers never reach the model
    event_branches: Tuple[str, ...] = ("averageInteractionsPerCrossing",
                                       "runNumber", "eventNumber")


# QA histogram binning: fixed edges so files can be filled one at a time
QA_BINS = {"pt_true": (np.geomspace(1.0, 5.0e3, 121),
                       r"Matched truth jet $p_\mathrm{T}$ [GeV]", True, True),
           "eta": (np.linspace(-5.0, 5.0, 101), r"Jet $\eta$", False, True),
           "phi": (np.linspace(-np.pi, np.pi, 65), r"Jet $\phi$", False, False),
           "mu": (np.arange(0.0, 301.0, 1.0), r"$\mu$", False, False)}


# ------------------------------------------------------------------- helpers

def _br(cfg: ProcessingConfig, name: str) -> str:
    return f"{cfg.calib}_{name}"


def _np(a: "ak.Array") -> np.ndarray:
    return ak.to_numpy(a).astype(np.float64)


def _pad(a: "ak.Array", n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Jagged (N, var) -> dense (N, n) values and validity mask."""
    padded = ak.pad_none(a, n, axis=-1, clip=True)
    return (ak.to_numpy(ak.fill_none(padded, 0.0)).astype(np.float32),
            ak.to_numpy(~ak.is_none(padded, axis=-1)).astype(np.float32))


def _jets(a: "ak.Array", keep: "ak.Array") -> "ak.Array":
    """(events, jets[, const]) -> selected jets, events flattened away."""
    return ak.flatten(a[keep], axis=1)


# -------------------------------------------------------------- book-keeping

@dataclass
class FileStats:
    """Selection book-keeping of one file (or of a merged production)."""

    name: str
    n_entries: int = 0
    n_jets: int = 0
    n_matched: int = 0     # jets with a usable truth match
    n_pass: int = 0        # jets passing the selection
    pt_true_max: float = float("nan")
    event_ids: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.uint64))
    hist: Dict[str, np.ndarray] = field(default_factory=dict)  # (2, nbins)

    @property
    def n_events(self) -> int:
        """Unique (runNumber, eventNumber) pairs."""
        return len(self.event_ids)

    def row(self) -> str:
        eff = 100 * self.n_pass / max(self.n_matched, 1)
        return (f"  {self.name[:34]:<36}{self.n_events:>11,}{self.n_jets:>11,}"
                f"{self.n_matched:>11,}{self.n_pass:>11,}{eff:>9.1f}"
                f"{self.pt_true_max:>11.1f}")


def file_stats(name: str, b: "ak.Array", keep: "ak.Array",
               cfg: ProcessingConfig) -> FileStats:
    """Counts and QA histograms of one file, before and after the selection."""
    sel = ak.to_numpy(ak.flatten(keep)).astype(bool)
    flat = lambda a: ak.to_numpy(ak.flatten(a)).astype(np.float64)
    mu = ak.broadcast_arrays(b["averageInteractionsPerCrossing"], b[_br(cfg, "pt")])[0]
    cols = {"pt_true": flat(b[_br(cfg, "matchedTruth_pt")]),
            "eta": flat(b[_br(cfg, "eta")]), "phi": flat(b[_br(cfg, "phi")]),
            "mu": flat(mu)}
    # unmatched jets have no target: excluded from every QA panel
    matched = np.isfinite(cols["pt_true"]) & (cols["pt_true"] > 0)
    good = matched & np.all([np.isfinite(c) for c in cols.values()], axis=0)
    hist = {}
    for key, col in cols.items():
        edges = QA_BINS[key][0]
        col = np.clip(col, edges[0], edges[-1])  # outer bins absorb overflow
        hist[key] = np.stack([np.histogram(col[good & sel], edges)[0],
                              np.histogram(col[good & ~sel], edges)[0]])
    ids = np.stack([ak.to_numpy(b["runNumber"]).astype(np.uint64),
                    ak.to_numpy(b["eventNumber"]).astype(np.uint64)], -1)
    return FileStats(name, n_entries=len(b), n_jets=len(sel),
                     n_matched=int(matched.sum()), n_pass=int(sel.sum()),
                     pt_true_max=float(cols["pt_true"][sel].max()) if sel.any()
                     else float("nan"),
                     event_ids=np.unique(ids, axis=0), hist=hist)


def merge_stats(stats: Sequence[FileStats]) -> FileStats:
    """Sum per-file book-keeping; events are re-uniqued across files."""
    return FileStats(
        "TOTAL", n_entries=sum(s.n_entries for s in stats),
        n_jets=sum(s.n_jets for s in stats),
        n_matched=sum(s.n_matched for s in stats),
        n_pass=sum(s.n_pass for s in stats),
        pt_true_max=float(np.nanmax([s.pt_true_max for s in stats])),
        event_ids=np.unique(np.concatenate([s.event_ids for s in stats]), axis=0),
        hist={k: sum(s.hist[k] for s in stats) for k in stats[0].hist})


def report_stats(stats: Sequence[FileStats], total: FileStats) -> None:
    """Print the production table and the jet accounting."""
    hdr = (f"  {'file':<36}{'events':>11}{'jets':>11}{'matched':>11}"
           f"{'selected':>11}{'eff[%]':>9}{'maxpT_t':>11}")
    line = "  " + "-" * (len(hdr) - 2)
    print("\n".join([hdr, line, *(s.row() for s in stats), line, total.row()]))
    pct = lambda k, n: f"{100 * k / max(n, 1):5.1f}%"
    print(f"[process] events: {total.n_events:,} unique from "
          f"{total.n_entries:,} entries")
    if total.n_entries != total.n_events:
        warnings.warn(f"{total.n_entries - total.n_events} duplicated events "
                      "across the inputs", UserWarning, stacklevel=2)
    print(f"[process] jets: {total.n_jets:,} total, {total.n_matched:,} matched, "
          f"{total.n_pass:,} selected ({pct(total.n_pass, total.n_matched)} of matched)")


# ---------------------------------------------------------------- QA figure

def plot_selection_qa(total: FileStats, out_dir: Union[str, Path],
                      cfg: ProcessingConfig) -> None:
    """2x2 stacked selected (bottom) / rejected (top) histograms + fraction."""
    import matplotlib.pyplot as plt
    from jetcal.style import FS_LEGEND, atlas_label, save

    fig = plt.figure(figsize=(20, 10))
    outer = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22)
    for i, (key, (edges, xlabel, logx, logy)) in enumerate(QA_BINS.items()):
        inner = outer[divmod(i, 2)].subgridspec(2, 1, height_ratios=[3, 1],
                                                hspace=0.05)
        ax = fig.add_subplot(inner[0])
        ax_f = fig.add_subplot(inner[1], sharex=ax)
        h = total.hist[key]
        nz = np.nonzero(h.sum(0))[0]  # trim empty outer bins
        if nz.size:
            h, edges = h[:, nz[0]:nz[-1] + 1], edges[nz[0]:nz[-1] + 2]
        sel, rej = h[0].astype(float), h[1].astype(float)
        ax.stairs(sel, edges, fill=True, color=_C_SEL, alpha=0.85,
                  label=f"Selected ({int(sel.sum()):,})")
        ax.stairs(sel + rej, edges, baseline=sel, fill=True, color=_C_REJ,
                  alpha=0.85, label=f"Rejected, matched ({int(rej.sum()):,})")
        ax.stairs(sel + rej, edges, color="black", lw=1.4)
        ax.set_ylabel("Jets / bin")
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(bottom=0.5)
        if key == "pt_true":
            ax.axvline(cfg.pt_cut, ls=":", color="black", lw=1.5)
        ax.legend(fontsize=FS_LEGEND - 2)
        ax.tick_params(labelbottom=False)
        if i == 0:
            atlas_label(ax)
        centers = np.sqrt(edges[:-1] * edges[1:]) if logx else 0.5 * (edges[:-1] + edges[1:])
        tot = sel + rej
        ax_f.plot(centers, np.where(tot > 0, sel / np.maximum(tot, 1), np.nan),
                  drawstyle="steps-mid", color="black", lw=1.6)
        ax_f.axhline(total.n_pass / max(total.n_matched, 1), ls="--",
                     color="gray", lw=1.4)
        ax_f.set_ylim(0.0, 1.05)
        ax_f.set_ylabel("Selected")
        ax_f.set_xlabel(xlabel)
        ax_f.set_xlim(edges[0], edges[-1])
        if logx:
            ax_f.set_xscale("log")
    save(fig, Path(out_dir) / f"{cfg.calib}_selection_qa")


# --------------------------------------------------------------- production

def iter_trees(raw_dir: Union[str, Path], tree_name: str
               ) -> Iterator[Tuple[Path, "uproot.TTree"]]:
    """``(path, TTree)`` for every ``.root`` file under ``raw_dir``."""
    paths = sorted(p for p in Path(raw_dir).rglob("*.root") if p.is_file())
    if not paths:
        raise FileNotFoundError(f"no .root files under {raw_dir}")
    for p in paths:
        yield p, uproot.open(p)[tree_name]


def read_arrays(tree: "uproot.TTree", cfg: ProcessingConfig) -> "ak.Array":
    """Read only the branches the production needs."""
    wanted = [_br(cfg, b) for b in cfg.branches] + list(cfg.event_branches)
    missing = [b for b in wanted if b not in set(tree.keys())]
    if missing:
        raise KeyError(f"tree is missing branches {missing}; is calib={cfg.calib!r} right?")
    return tree.arrays(wanted, library="ak")


def jet_selection(b: "ak.Array", cfg: ProcessingConfig) -> "ak.Array":
    """Per-jet mask: pT_EM > 0, matched truth pT > cut, >= 1 constituent."""
    return ((b[_br(cfg, "pt")] > 0) & (b[_br(cfg, "matchedTruth_pt")] > cfg.pt_cut)
            & (b[_br(cfg, "nConstituents")] > 0))


def build_features(b: "ak.Array", keep: "ak.Array", cfg: ProcessingConfig) -> dict:
    """Selected jets -> one dataset part (un-normalized)."""
    sel = lambda name: _jets(b[_br(cfg, name)], keep)
    pt_j, eta_j, phi_j = sel("pt"), sel("eta"), sel("phi")  # awkward: broadcast
    pt, eta, m = _np(pt_j), _np(eta_j), _np(sel("m"))       # numpy: jet columns
    c_pt, c_eta, c_phi = (sel(k) for k in ("constituentPt", "constituentEta",
                                           "constituentPhi"))
    n_max = ak.max(ak.num(c_pt, axis=-1))
    if n_max is not None and n_max > cfg.max_const:
        warnings.warn(f"jets with {n_max} constituents exceed max_const="
                      f"{cfg.max_const}; clusters are unordered, so truncation "
                      "drops random ones", UserWarning, stacklevel=2)

    d_phi = (c_phi - phi_j + np.pi) % (2 * np.pi) - np.pi
    cols = [c_pt, c_eta, c_phi, np.log(np.maximum(c_pt, 1e-3)),
            np.log(np.maximum(c_pt / pt_j, 1e-6)), c_eta - eta_j,
            np.sin(d_phi), np.cos(d_phi)]
    padded = [_pad(c, cfg.max_const) for c in cols]
    mask = padded[0][1]
    x_phi = np.stack([p for p, _ in padded], -1) * mask[..., None]
    x_glob = np.stack([np.log(pt), eta, np.abs(eta), np.log1p(m / pt)], -1)
    pt_true = _np(sel("matchedTruth_pt"))
    return {"x_phi": x_phi, "mask": mask, "x_glob": x_glob.astype(np.float32),
            "y": np.log(pt_true / pt)[:, None].astype(np.float32),
            "pt_reco": pt.astype(np.float32), "eta_reco": eta.astype(np.float32),
            "pt_true": pt_true.astype(np.float32),
            "eta_true": _np(sel("matchedTruth_eta")).astype(np.float32)}


def produce_datasets(raw_dir: Union[str, Path], out_dir: Union[str, Path],
                     cfg: ProcessingConfig) -> Dict[str, Path]:
    """ROOT files -> ``<calib>_train.npz`` / ``<calib>_test.npz``.

    Normalization is fitted on the train split only (constituent statistics
    over real clusters only) and applied to both splits.
    """
    t0 = time()
    print(f"[process] calib={cfg.calib} pt_cut={cfg.pt_cut} seed={cfg.seed}")
    parts, stats = [], []
    for path, tree in iter_trees(raw_dir, cfg.tree_name):
        b = read_arrays(tree, cfg)
        keep = jet_selection(b, cfg)
        stats.append(file_stats(path.name, b, keep, cfg))
        parts.append(build_features(b, keep, cfg))
    total = merge_stats(stats)
    report_stats(stats, total)
    if total.n_pass == 0:
        raise RuntimeError("the selection kept 0 jets; check --calib and --pt-cut")
    if cfg.qa_plots:
        plot_selection_qa(total, Path(out_dir) / "qa", cfg)

    data = {**concat_datasets(parts), "phi_lbls": list(PHI_LABELS),
            "glob_lbls": list(GLOB_LABELS), "y_lbls": list(Y_LABELS)}
    train, test = shuffle_split(data, cfg.test_frac, cfg.seed)
    norm = {"phi": fit_norm(train["x_phi"], train["mask"]),
            "glob": fit_norm(train["x_glob"]), "y": fit_norm(train["y"])}
    paths = {}
    for split, d in (("train", train), ("test", test)):
        d["x_phi"] = apply_norm(d["x_phi"], norm["phi"], d["mask"])
        d["x_glob"] = apply_norm(d["x_glob"], norm["glob"])
        d["norm"] = norm
        d["meta"] = str(dict(calib=cfg.calib, split=split, seed=cfg.seed,
                             pt_cut=cfg.pt_cut, max_const=cfg.max_const,
                             n_events=total.n_events, n=len(d["y"])))
        validate_dataset(d)
        paths[split] = save_dataset(Path(out_dir) / f"{cfg.calib}_{split}.npz", d)
        print(f"[process] wrote {paths[split]} ({len(d['y']):,} jets)")
    print(f"[process] done in {time() - t0:.0f}s")
    return paths
