# Philippe Joly, ATLAS Level-0 trigger ML
"""Figure conventions shared by every plotting module.

    * Ideal reference (R = 1, the diagonal, the float model) -> dashed grey.
    * "Uncalibrated" -> dashed black; "Binned median" -> solid orange.
    * Binned median inside a 2D map -> solid red; its 16/84% band -> dotted blue.
    * Every binned axis states its bin widths (:func:`annotate_bins`).
    * Every figure is written as SVG + 400 dpi PNG (:func:`save`).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot
import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402
import numpy as np  # noqa: E402

ATLAS_TEXT = "Simulation Work in Progress"
SAVE_FORMATS = ("svg", "png")
SAVE_DPI = 400

LW, LW_REF, MS = 2.6, 2.2, 8.0
FS_LABEL, FS_TICK, FS_LEGEND, FS_ANNOT = 26, 20, 19, 17

IDEAL_STYLE = dict(ls="--", color="gray", lw=LW_REF)
MEDIAN_STYLE = dict(ls="-", color="red", lw=LW)
IQR_STYLE = dict(ls=":", color="blue", lw=LW_REF)
C_BINNED_MEDIAN, C_UNCALIBRATED = "#e69f00", "black"

_REFERENCE_STYLE = {"Uncalibrated": dict(color=C_UNCALIBRATED, ls="--"),
                    "Binned median": dict(color=C_BINNED_MEDIAN, ls="-")}
# colour-blind-safe model colours; orange is reserved for the baseline
MODEL_COLORS = ("#0072b2", "#009e73", "#cc79a7", "#56b4e9", "#d55e00",
                "#f0e442", "#882255", "#44aa99")

_RC = {
    "savefig.dpi": SAVE_DPI, "savefig.bbox": "tight", "font.size": 21,
    "axes.labelsize": FS_LABEL, "axes.titlesize": FS_LABEL,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND, "legend.frameon": False,
    "lines.linewidth": LW, "lines.markersize": MS,
    "lines.markeredgewidth": 1.6, "axes.linewidth": 2.0,
    "xtick.major.width": 2.0, "ytick.major.width": 2.0,
    "xtick.minor.width": 1.4, "ytick.minor.width": 1.4,
    "xtick.major.size": 11, "ytick.major.size": 11,
    "xtick.minor.size": 6, "ytick.minor.size": 6, "errorbar.capsize": 0,
}


def apply_style() -> None:
    """ATLAS style plus publication sizing (called on import)."""
    plt.style.use(hep.style.ATLAS)
    plt.rcParams.update(_RC)


apply_style()


def series_styles(labels: Iterable[str]) -> Dict[str, dict]:
    """Line style per series: fixed for the two references, cycled for models."""
    styles, j = {}, 0
    for lab in labels:
        if lab in _REFERENCE_STYLE:
            styles[lab] = dict(_REFERENCE_STYLE[lab], lw=LW)
        else:
            styles[lab] = dict(color=MODEL_COLORS[j % len(MODEL_COLORS)],
                               ls="-", lw=LW)
            j += 1
    if "Uncalibrated" in styles:
        styles["Uncalibrated"]["zorder"] = 5
    return styles


def atlas_label(ax: plt.Axes, fontsize: int = FS_ANNOT) -> None:
    """ATLAS Simulation Work in Progress label, top left."""
    hep.atlas.text(ATLAS_TEXT, ax=ax, loc=0, fontsize=fontsize)


# ------------------------------------------------------------- LaTeX labels

_ATOM = {"pt": r"p_\mathrm{T}", "eta": r"\eta", "phi": r"\phi", "m": r"m",
         "z": r"z", "mu": r"\langle\mu\rangle", "deta": r"\Delta\eta",
         "dphi": r"\Delta\phi", "e": r"E"}
_QUALIFIER = {"reco", "true", "calib", "c", "em", "quant", "float"}
_QUAL_TEXT = {"c": "const", "em": "EM"}


def _tex(name: str) -> str:
    """Math body for a pipeline label, e.g. ``log_pt_c`` -> log(pT^const)."""
    tokens = [t.lower() for t in re.split(r"[_\s.]+", name.strip()) if t]
    ops = []
    while tokens and tokens[0] in ("log", "abs", "sin", "cos"):
        ops.append(tokens.pop(0))
    if "over" in tokens:  # ratio, e.g. m_over_pt
        i = tokens.index("over")
        body = f"{_tex('_'.join(tokens[:i]))} / {_tex('_'.join(tokens[i + 1:]))}"
    elif tokens:
        body = _ATOM.get(tokens[0], r"\mathrm{%s}" % tokens[0])
        quals = [_QUAL_TEXT.get(t, t) for t in tokens[1:]]
        if quals:
            body += r"^{\mathrm{%s}}" % ",".join(quals)
    else:
        body = ""
    for op in reversed(ops):
        body = f"|{body}|" if op == "abs" else rf"\{op}({body})"
    return body


def tex_label(name: str, unit: str = "") -> str:
    """LaTeX label for a variable; a ``level:`` prefix is stripped."""
    out = f"${_tex(str(name).split(':', 1)[-1])}$"
    return f"{out} [{unit}]" if unit else out


def tex_labels(names: Iterable[str]) -> List[str]:
    """:func:`tex_label` over an iterable (tick labels)."""
    return [tex_label(n) for n in names]


# ------------------------------------------------------------ axis helpers

def bin_width_note(edges: Sequence[float], unit: str = "",
                   name: Optional[str] = None) -> str:
    """``"<name> bins: N, width w"`` (or a width range for adaptive bins)."""
    w = np.diff(np.asarray(edges, np.float64))
    u = f" {unit}" if unit else ""
    head = f"{name} bins" if name else "bins"
    if np.allclose(w, w[0], rtol=1e-3, atol=0.0):
        return f"{head}: {w.size}, width {w[0]:.3g}{u}"
    return f"{head}: {w.size}, width {w.min():.3g}–{w.max():.3g}{u}"


def annotate_bins(ax: plt.Axes, *notes: str, loc: str = "upper left") -> None:
    """Write bin-width notes inside an axes (one line per binned axis)."""
    x, y, ha, va = {"upper left": (0.02, 0.98, "left", "top"),
                    "upper right": (0.98, 0.98, "right", "top"),
                    "lower left": (0.02, 0.02, "left", "bottom"),
                    "lower right": (0.98, 0.02, "right", "bottom")}[loc]
    ax.text(x, y, "\n".join(n for n in notes if n), transform=ax.transAxes,
            ha=ha, va=va, fontsize=FS_ANNOT, color="0.25", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=2.5))


def draw_ideal(ax: plt.Axes, kind: str = "h", value: float = 1.0,
               xy: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
               label: Optional[str] = None) -> None:
    """Ideal reference line: ``"h"``/``"v"`` at ``value`` or ``"diag"`` through ``xy``."""
    style = dict(IDEAL_STYLE, **({"label": label} if label else {}))
    if kind == "h":
        ax.axhline(value, **style)
    elif kind == "v":
        ax.axvline(value, **style)
    else:
        ax.plot(xy[0], xy[1], **style)


def tidy_log_axis(ax: plt.Axes, which: str = "x") -> None:
    """Label 1/2/5 per decade on a log axis; silence minor labels."""
    axis = ax.xaxis if which == "x" else ax.yaxis
    tk = matplotlib.ticker
    axis.set_major_locator(tk.LogLocator(10.0, subs=(1.0, 2.0, 5.0), numticks=15))
    axis.set_minor_locator(tk.LogLocator(10.0, subs=tuple(np.arange(1, 10) * 0.1),
                                         numticks=60))
    axis.set_major_formatter(tk.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(tk.NullFormatter())


def save(fig: plt.Figure, path: Optional[Union[str, Path]]) -> List[Path]:
    """Write ``fig`` as SVG + PNG next to ``path`` (suffix ignored), then close."""
    if path is None:
        plt.close(fig)
        return []
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.stem if path.suffix.lower() in (".png", ".svg", ".pdf") else path.name
    written = []
    for ext in SAVE_FORMATS:
        out = path.parent / f"{stem}.{ext}"
        fig.savefig(out, dpi=SAVE_DPI, bbox_inches="tight", format=ext)
        written.append(out)
    print(f"[plot] saved {path.parent / stem}.{{{','.join(SAVE_FORMATS)}}}")
    plt.close(fig)
    return written
