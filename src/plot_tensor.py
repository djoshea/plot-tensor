"""
pt — Plot Tensor
================
Visualize multi-dimensional arrays as faceted seaborn line plots.

Entry points
------------
``pt.line(tensor, ...)``  — faceted line plot (see :func:`line` for full API)

Supported tensor types
----------------------
``numpy.ndarray``, JAX ``Array``, Penzai ``NamedArray``, ``xarray.DataArray``
"""
from __future__ import annotations

import colorsys
import itertools
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Public type aliases
# ---------------------------------------------------------------------------
AxisRef = Union[int, str]
AxisRefs = Union[AxisRef, List[AxisRef]]

# Axis names that auto-bind to the x/time role if time= is not supplied
_AUTO_TIME_NAMES: frozenset[str] = frozenset({"time", "t", "T", "x"})

# Default linestyle cycle for the style= channel
_DEFAULT_LINESTYLES: List[Any] = [
    "-",
    "--",
    ":",
    "-.",
    (0, (3, 1, 1, 1)),       # dash-dot-dot
    (0, (5, 1)),              # dense dash
]


# ---------------------------------------------------------------------------
# 1. Array normalisation
# ---------------------------------------------------------------------------

def _extract_array_info(
    tensor,
    dim_names: Optional[List],
    coords: Optional[Dict],
) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
    """
    Normalize *tensor* to ``(numpy_array, axis_names, coord_map)``.

    ``coord_map`` maps each axis name to a 1-D coordinate array.
    Missing coords default to integer ranges.

    *dim_names* may contain plain strings **or** ``(name, labels)`` tuples.
    *coords* is a ``{name_or_index: labels}`` dict that is merged on top.
    Either ``None`` or ``()`` as a label value triggers ``np.arange(n)``.
    """
    coord_map: Dict[str, np.ndarray] = {}

    # --- xarray DataArray ---
    try:
        import xarray as xr  # type: ignore
        if isinstance(tensor, xr.DataArray):
            if dim_names is None:
                dim_names = list(tensor.dims)
            for dim in tensor.dims:
                if dim in tensor.coords:
                    coord_map[dim] = np.asarray(tensor.coords[dim].values)
            tensor = np.asarray(tensor.values)
    except ImportError:
        pass

    # --- Penzai NamedArray ---
    try:
        from penzai.core import named_axes as _pz_na  # type: ignore
        if isinstance(tensor, _pz_na.NamedArray):
            axis_names_pz = list(tensor.named_shape.keys())
            if dim_names is None:
                dim_names = axis_names_pz
            tensor = np.asarray(tensor.unwrap(*axis_names_pz))
    except (ImportError, AttributeError):
        pass

    # --- JAX array ---
    try:
        import jax  # type: ignore
        if isinstance(tensor, jax.Array):
            tensor = np.asarray(tensor)
    except ImportError:
        pass

    arr = np.asarray(tensor)
    ndim = arr.ndim

    # Parse dim_names: may be list of str or list of (str, labels)
    names: List[str] = []
    if dim_names is not None:
        if len(dim_names) != ndim:
            raise ValueError(
                f"dim_names has {len(dim_names)} entries but array has {ndim} dims."
            )
        for entry in dim_names:
            if isinstance(entry, tuple):
                name, labels = entry[0], entry[1] if len(entry) > 1 else None
                names.append(name)
                if labels is not None and len(labels) > 0:
                    coord_map[name] = np.asarray(labels)
            else:
                names.append(str(entry))
    else:
        names = [f"dim_{i}" for i in range(ndim)]

    # Merge user-supplied coords dict
    if coords is not None:
        for key, vals in coords.items():
            resolved = names[key] if isinstance(key, int) else str(key)
            if resolved not in names:
                raise ValueError(
                    f"coords key {key!r} not found in axis names {names}."
                )
            if vals is None or (hasattr(vals, '__len__') and len(vals) == 0):
                pass  # will be filled with arange below
            else:
                coord_map[resolved] = np.asarray(vals)

    # Fill any still-missing coords with integer ranges
    for name, size in zip(names, arr.shape):
        if name not in coord_map:
            coord_map[name] = np.arange(size)

    return arr, names, coord_map


# ---------------------------------------------------------------------------
# 2. Axis reference resolution
# ---------------------------------------------------------------------------

def _resolve_axis(ref: AxisRef, names: List[str]) -> int:
    if isinstance(ref, int):
        if not (0 <= ref < len(names)):
            raise ValueError(f"Axis index {ref} out of range (ndim={len(names)}).")
        return ref
    if ref not in names:
        raise ValueError(f"Axis name {ref!r} not found. Available: {names}.")
    return names.index(ref)


def _resolve_axes(refs: AxisRefs, names: List[str]) -> List[int]:
    if isinstance(refs, (int, str)):
        return [_resolve_axis(refs, names)]
    return [_resolve_axis(r, names) for r in refs]


# ---------------------------------------------------------------------------
# 3. Role assignment
# ---------------------------------------------------------------------------

@dataclass
class _RoleMap:
    """Resolved per-axis roles after parsing all kwargs."""
    ndim: int
    orig_names: List[str]
    orig_shapes: Tuple[int, ...]
    coord_map: Dict[str, np.ndarray]

    time_idx: int
    hue_idxs: List[int]        # 1+ axes for linearised hue/color
    color2d_idxs: List[int]    # exactly 2 axes for 2-D palette (or empty)
    style_idx: Optional[int]
    size_idxs: List[int]
    row_idx: Optional[int]
    col_idx: Optional[int]
    reduced_idxs: List[int]    # axes that will be mean-reduced


def _assign_roles(
    arr: np.ndarray,
    names: List[str],
    coord_map: Dict[str, np.ndarray],
    *,
    time: Optional[AxisRef],
    x: Optional[AxisRef],
    hue: Optional[AxisRefs],
    color: Optional[AxisRefs],
    color2d: Optional[List[AxisRef]],
    style: Optional[AxisRef],
    size: Optional[AxisRefs],
    row: Optional[AxisRef],
    col: Optional[AxisRef],
) -> _RoleMap:
    ndim = arr.ndim
    shapes = tuple(arr.shape)

    # --- Alias resolution ---
    if time is not None and x is not None:
        raise ValueError("Specify time= or x= but not both.")
    x_ref = time if time is not None else x

    if hue is not None and color is not None:
        raise ValueError("Specify hue= or color= but not both (they are aliases).")
    hue_ref = hue if hue is not None else color

    if hue_ref is not None and color2d is not None:
        raise ValueError("Specify hue=/color= OR color2d= but not both.")

    # --- Resolve each role ---
    time_idx: Optional[int] = None
    if x_ref is not None:
        time_idx = _resolve_axis(x_ref, names)
    else:
        # Auto-detect from reserved names
        for i, nm in enumerate(names):
            if nm in _AUTO_TIME_NAMES:
                time_idx = i
                break
        if time_idx is None:
            time_idx = ndim - 1  # fallback: last axis

    hue_idxs: List[int] = _resolve_axes(hue_ref, names) if hue_ref is not None else []

    if color2d is not None:
        color2d_idxs = _resolve_axes(color2d, names)
        if len(color2d_idxs) != 2:
            raise ValueError(
                f"color2d= requires exactly 2 axes, got {len(color2d_idxs)}."
            )
    else:
        color2d_idxs = []

    style_idx: Optional[int] = _resolve_axis(style, names) if style is not None else None
    size_idxs: List[int] = _resolve_axes(size, names) if size is not None else []
    row_idx: Optional[int] = _resolve_axis(row, names) if row is not None else None
    col_idx: Optional[int] = _resolve_axis(col, names) if col is not None else None

    # --- Validate: no axis assigned to two roles ---
    all_idxs = (
        [time_idx]
        + hue_idxs
        + color2d_idxs
        + ([style_idx] if style_idx is not None else [])
        + size_idxs
        + ([row_idx] if row_idx is not None else [])
        + ([col_idx] if col_idx is not None else [])
    )
    seen: set = set()
    for idx in all_idxs:
        if idx in seen:
            raise ValueError(
                f"Axis {names[idx]!r} (index {idx}) is assigned to more than one role."
            )
        seen.add(idx)

    # --- Smart defaults for 2-D arrays ---
    unassigned = [i for i in range(ndim) if i not in seen]
    if unassigned and ndim == 2 and not hue_idxs and not color2d_idxs:
        # Promote the non-time axis to hue automatically
        hue_idxs = [i for i in unassigned if i != time_idx]
        seen.update(hue_idxs)
        unassigned = [i for i in range(ndim) if i not in seen]

    # --- Mean-reduce anything still unassigned ---
    reduced_idxs: List[int] = []
    if unassigned:
        unassigned_names = [names[i] for i in unassigned]
        warnings.warn(
            f"Axes {unassigned_names} are not assigned to any channel and will be "
            "mean-reduced.  Pass them explicitly to suppress this warning.",
            UserWarning,
            stacklevel=3,
        )
        reduced_idxs = unassigned

    return _RoleMap(
        ndim=ndim,
        orig_names=names,
        orig_shapes=shapes,
        coord_map=coord_map,
        time_idx=time_idx,
        hue_idxs=hue_idxs,
        color2d_idxs=color2d_idxs,
        style_idx=style_idx,
        size_idxs=size_idxs,
        row_idx=row_idx,
        col_idx=col_idx,
        reduced_idxs=reduced_idxs,
    )


# ---------------------------------------------------------------------------
# 4. Verbose role table
# ---------------------------------------------------------------------------

def _print_role_table(rm: _RoleMap) -> None:
    def _fmt_coords(arr: np.ndarray) -> str:
        n = len(arr)
        if n == 0:
            return "[]"
        if n <= 4:
            return "[" + ", ".join(str(v) for v in arr) + "]"
        return f"[{arr[0]} .. {arr[-1]}]  ({n})"

    def _role_name(i: int) -> str:
        if i == rm.time_idx:
            return "x-axis"
        if i in rm.hue_idxs:
            return "hue" if len(rm.hue_idxs) == 1 else f"hue[{rm.hue_idxs.index(i)}]"
        if i in rm.color2d_idxs:
            return f"color2d[{rm.color2d_idxs.index(i)}]"
        if i == rm.style_idx:
            return "style"
        if i in rm.size_idxs:
            return "size" if len(rm.size_idxs) == 1 else f"size[{rm.size_idxs.index(i)}]"
        if i == rm.row_idx:
            return "row"
        if i == rm.col_idx:
            return "col"
        if i in rm.reduced_idxs:
            return "mean-reduced"
        return "?"

    hdr = f"{'dim':<12}  {'shape':>5}  {'role':<14}  coords"
    sep = f"{'-'*12}  {'-'*5}  {'-'*14}  {'-'*24}"
    print(hdr)
    print(sep)
    for i, (name, size) in enumerate(zip(rm.orig_names, rm.orig_shapes)):
        role = _role_name(i)
        coords = _fmt_coords(rm.coord_map[name])
        print(f"{name:<12}  {size:>5}  {role:<14}  {coords}")
    print()


# ---------------------------------------------------------------------------
# 5. DataFrame construction
# ---------------------------------------------------------------------------

def _apply_reductions(
    arr: np.ndarray,
    names: List[str],
    coord_map: Dict[str, np.ndarray],
    reduced_idxs: List[int],
) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
    """Mean-reduce *arr* along each index in *reduced_idxs* (sorted descending)."""
    for idx in sorted(reduced_idxs, reverse=True):
        arr = arr.mean(axis=idx)
    keep = [i for i in range(len(names)) if i not in reduced_idxs]
    new_names = [names[i] for i in keep]
    new_coords = {names[i]: coord_map[names[i]] for i in keep}
    return arr, new_names, new_coords


def _build_dataframe(
    arr: np.ndarray,
    names: List[str],
    coord_map: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Long-form DataFrame: one col per axis (coordinate-labelled) + ``value``."""
    grids = np.meshgrid(
        *[np.arange(arr.shape[i]) for i in range(arr.ndim)],
        indexing="ij",
    )
    data: Dict[str, np.ndarray] = {}
    for name, grid in zip(names, grids):
        data[name] = coord_map[name][grid.ravel()]
    data["value"] = arr.ravel()
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# 6. Colour helpers
# ---------------------------------------------------------------------------

def _rgba(c) -> Tuple[float, float, float, float]:
    """Coerce any matplotlib colour spec to an RGBA tuple."""
    c = tuple(float(v) for v in c)
    return (c[0], c[1], c[2], 1.0) if len(c) == 3 else (c[0], c[1], c[2], c[3])


def _make_hue_palette(n: int, palette: Optional[Any]) -> List[Tuple]:
    """Return *n* RGBA colours. Default: husl (≤12) or viridis (>12)."""
    if palette is not None:
        if isinstance(palette, str):
            raw = sns.color_palette(palette, n)
        else:
            raw = list(palette)[:n]
            if len(raw) < n:
                raw = sns.color_palette(palette, n)
    elif n <= 12:
        raw = sns.color_palette("husl", n)
    else:
        raw = sns.color_palette("viridis", n)
    return [_rgba(c) for c in raw]


def _make_color2d_palette(n1: int, n2: int) -> np.ndarray:
    """
    Return an ``(n1, n2, 4)`` RGBA grid.
    Axis 0 → hue (0.05–0.85), axis 1 → lightness (0.35–0.65), saturation 0.75.
    """
    out = np.zeros((n1, n2, 4), dtype=float)
    for i in range(n1):
        hue = (0.05 + i / max(n1 - 1, 1) * 0.80) if n1 > 1 else 0.45
        for j in range(n2):
            lightness = (0.35 + j / max(n2 - 1, 1) * 0.30) if n2 > 1 else 0.50
            r, g, b = colorsys.hls_to_rgb(hue, lightness, 0.75)
            out[i, j] = [r, g, b, 1.0]
    return out


# ---------------------------------------------------------------------------
# 7. Size and style helpers
# ---------------------------------------------------------------------------

def _make_linewidths(
    n: int,
    sizes: Tuple[float, float],
    sqrt_mode: bool,
) -> np.ndarray:
    """Linewidth array of length *n* spanning ``sizes``."""
    lo, hi = float(sizes[0]), float(sizes[1])
    if n == 1:
        return np.array([(lo + hi) / 2.0])
    if sqrt_mode:
        return np.sqrt(np.linspace(lo ** 2, hi ** 2, n))
    return np.linspace(lo, hi, n)


def _make_style_linestyles(n: int, dashes: Optional[Any]) -> List[Any]:
    """Linestyle list of length *n*."""
    if dashes is not None:
        # Accept a list/dict of explicit patterns
        if isinstance(dashes, dict):
            patterns = list(dashes.values())
        else:
            patterns = list(dashes)
        # Cycle if fewer patterns than needed
        return [patterns[i % len(patterns)] for i in range(n)]
    return [_DEFAULT_LINESTYLES[i % len(_DEFAULT_LINESTYLES)] for i in range(n)]


# ---------------------------------------------------------------------------
# 8. Legend helpers
# ---------------------------------------------------------------------------

def _patch_legend(
    ax: plt.Axes,
    labels: List[str],
    colors: List[Tuple],
    title: str,
) -> None:
    handles = [mpatches.Patch(color=c, label=lb) for lb, c in zip(labels, colors)]
    ax.legend(handles=handles, title=title, bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize="small", title_fontsize="small", framealpha=0.9)


def _linewidth_legend(
    ax: plt.Axes,
    labels: List[str],
    widths: List[float],
    title: str,
) -> None:
    handles = [
        mlines.Line2D([], [], color="0.3", linewidth=lw, label=lb)
        for lb, lw in zip(labels, widths)
    ]
    ax.legend(handles=handles, title=title, bbox_to_anchor=(1.02, 0), loc="lower left",
              fontsize="small", title_fontsize="small", framealpha=0.9)


def _style_legend(
    ax: plt.Axes,
    labels: List[str],
    linestyles: List[Any],
    title: str,
) -> None:
    handles = [
        mlines.Line2D([], [], color="0.3", linestyle=ls, linewidth=1.5, label=lb)
        for lb, ls in zip(labels, linestyles)
    ]
    ax.legend(handles=handles, title=title, bbox_to_anchor=(1.02, 0.5), loc="center left",
              fontsize="small", title_fontsize="small", framealpha=0.9)


def _color2d_legend(
    fig: plt.Figure,
    grid: np.ndarray,
    ax_names: List[str],
    ax_labels: List[List],
) -> None:
    """Inset swatch grid to the right of the figure."""
    n1, n2 = grid.shape[:2]
    inset = fig.add_axes([0.92, 0.15, 0.055, 0.70])
    inset.set_xlim(-0.5, n2 - 0.5)
    inset.set_ylim(-0.5, n1 - 0.5)
    for i in range(n1):
        for j in range(n2):
            inset.add_patch(
                mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1.0, 1.0,
                    color=tuple(grid[i, j]),
                    linewidth=0,
                )
            )
    inset.set_xticks(range(n2))
    inset.set_xticklabels(
        [str(v) for v in ax_labels[1]], fontsize=7, rotation=45, ha="right"
    )
    inset.set_yticks(range(n1))
    inset.set_yticklabels([str(v) for v in ax_labels[0]], fontsize=7)
    inset.set_xlabel(ax_names[1], fontsize=8)
    inset.set_ylabel(ax_names[0], fontsize=8)
    inset.set_title("color", fontsize=8)
    inset.tick_params(length=0)


# ---------------------------------------------------------------------------
# 9. Main entry point
# ---------------------------------------------------------------------------

def line(
    tensor,
    *,
    # x-axis
    time: Optional[AxisRef] = None,
    x: Optional[AxisRef] = None,
    # colour channels  (specify hue/color OR color2d, not both)
    hue: Optional[AxisRefs] = None,
    color: Optional[AxisRefs] = None,      # alias for hue
    color2d: Optional[List[AxisRef]] = None,
    # other aesthetic channels
    style: Optional[AxisRef] = None,
    size: Optional[AxisRefs] = None,
    # facets
    row: Optional[AxisRef] = None,
    col: Optional[AxisRef] = None,
    # axis metadata
    dim_names: Optional[List] = None,
    coords: Optional[Dict] = None,
    # palette / style control
    palette: Optional[Any] = None,
    sizes: Tuple[float, float] = (0.5, 2.5),
    scale_linewidth_sqrt: bool = False,
    dashes: Optional[Any] = None,
    # figure
    height: float = 3.0,
    aspect: float = 1.5,
    col_wrap: Optional[int] = None,
    alpha: float = 0.8,
    legend: bool = True,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    verbose: bool = False,
    **kwargs: Any,
) -> sns.FacetGrid:
    """
    Render a multi-dimensional tensor as a faceted seaborn line plot.

    Parameters
    ----------
    tensor:
        Input data.  Accepts ``numpy.ndarray``, JAX ``Array``,
        Penzai ``NamedArray``, ``xarray.DataArray``.
    time, x:
        Axis for the x-axis (aliases; specify one or neither).
        Auto-detected from axis names ``{'time','t','T','x'}``; falls back
        to the last axis.
    hue, color:
        Axis/axes for line colour — aliases, specify at most one.
        Single axis → sequential 1-D palette.
        Multiple axes → Cartesian-product linearised palette.
    color2d:
        List of **exactly 2** axes for a 2-D HLS palette:
        first axis → hue (colour wheel), second → lightness.
        Mutually exclusive with ``hue``/``color``.
    style:
        Single axis for linestyle (cycles solid → dashed → dotted → …).
    size:
        Axis/axes for linewidth (linearised across Cartesian product).
    row, col:
        Axes for facet rows/columns.
    dim_names:
        Names for each axis (positional).  Each entry may be a plain string
        ``'name'`` or a ``('name', coord_array)`` tuple.
        Auto-detected from ``NamedArray`` / ``xarray.DataArray``.
    coords:
        ``{name_or_index: label_array}`` — coordinate values per axis.
        ``None`` / ``()`` → integer range.  Merged with *dim_names* labels.
    palette:
        Seaborn/matplotlib palette name or list of colours for the ``hue``
        channel.  Defaults to ``'husl'`` (n ≤ 12) or ``'viridis'`` (n > 12).
    sizes:
        ``(min_lw, max_lw)`` linewidth range in points.
    scale_linewidth_sqrt:
        If ``True``, map size to sqrt-scaled linewidth (area-proportional).
    dashes:
        Override the linestyle cycle: list of matplotlib dash specs, or a
        dict mapping axis values to dash specs.
    height:
        Height of each facet in inches.
    aspect:
        Aspect ratio (width / height) of each facet.
    col_wrap:
        Wrap columns at this count (only when ``row`` is ``None``).
    alpha:
        Line opacity (0–1).
    legend:
        Show colour / size / style legends.  Set ``False`` to suppress.
    title, xlabel, ylabel:
        Figure title and axis labels.
    verbose:
        Print a table describing how each axis is mapped.
    **kwargs:
        Additional keyword arguments forwarded to ``ax.plot()``.

    Returns
    -------
    seaborn.FacetGrid

    Examples
    --------
    >>> import numpy as np, plot_tensor as pt
    >>> # 1-D: single line (time auto-detected)
    >>> pt.line(np.random.randn(100).cumsum())

    >>> # 3-D named dims, explicit roles
    >>> x = np.random.randn(4, 6, 100).cumsum(-1)
    >>> g = pt.line(x, dim_names=['batch','layer','time'],
    ...             hue='layer', row='batch', verbose=True)

    >>> # 4-D with 2-D colour palette
    >>> x = np.random.randn(3, 4, 5, 50)
    >>> g = pt.line(x, dim_names=['run','head','layer','t'],
    ...             color2d=['head','layer'], col='run')
    """
    # ------------------------------------------------------------------
    # Step 1 — normalise input
    # ------------------------------------------------------------------
    arr, names, coord_map = _extract_array_info(tensor, dim_names, coords)
    if arr.ndim == 0:
        raise ValueError("Cannot plot a scalar (0-d) tensor.")

    # ------------------------------------------------------------------
    # Step 2 — assign roles, detect unassigned axes
    # ------------------------------------------------------------------
    rm = _assign_roles(
        arr, names, coord_map,
        time=time, x=x,
        hue=hue, color=color, color2d=color2d,
        style=style, size=size,
        row=row, col=col,
    )

    if verbose:
        _print_role_table(rm)

    # ------------------------------------------------------------------
    # Step 3 — mean-reduce unassigned axes, rebuild names/coords
    # ------------------------------------------------------------------
    if rm.reduced_idxs:
        arr, names, coord_map = _apply_reductions(
            arr, names, coord_map, rm.reduced_idxs
        )
        # Re-index everything in rm relative to the reduced array
        def _reindex(old_idx: Optional[int]) -> Optional[int]:
            if old_idx is None:
                return None
            removed_before = sum(1 for r in rm.reduced_idxs if r < old_idx)
            return old_idx - removed_before

        time_idx   = _reindex(rm.time_idx)
        hue_idxs   = [_reindex(i) for i in rm.hue_idxs]
        c2d_idxs   = [_reindex(i) for i in rm.color2d_idxs]
        style_idx  = _reindex(rm.style_idx)
        size_idxs  = [_reindex(i) for i in rm.size_idxs]
        row_idx    = _reindex(rm.row_idx)
        col_idx    = _reindex(rm.col_idx)
    else:
        time_idx   = rm.time_idx
        hue_idxs   = rm.hue_idxs
        c2d_idxs   = rm.color2d_idxs
        style_idx  = rm.style_idx
        size_idxs  = rm.size_idxs
        row_idx    = rm.row_idx
        col_idx    = rm.col_idx

    time_name = names[time_idx]

    # ------------------------------------------------------------------
    # Step 4 — build long-form DataFrame
    # ------------------------------------------------------------------
    df = _build_dataframe(arr, names, coord_map)

    # ------------------------------------------------------------------
    # Step 5 — colour lookup columns
    # ------------------------------------------------------------------
    hue_names   = [names[i] for i in hue_idxs]
    c2d_names   = [names[i] for i in c2d_idxs]
    color2d_grid: Optional[np.ndarray] = None

    if hue_idxs:
        if len(hue_idxs) == 1:
            cname = hue_names[0]
            unique_vals = coord_map[cname].tolist()
            pal = _make_hue_palette(len(unique_vals), palette)
            hue_cmap: Dict[Any, Tuple] = dict(zip(unique_vals, pal))
            df["_color"] = df[cname].map(hue_cmap)
        else:
            all_vals = [coord_map[n].tolist() for n in hue_names]
            combos = list(itertools.product(*all_vals))
            pal = _make_hue_palette(len(combos), palette)
            lin_cmap: Dict[Any, Tuple] = dict(zip(combos, pal))
            df["_color"] = [
                lin_cmap[tuple(row[n] for n in hue_names)]
                for _, row in df[hue_names].iterrows()
            ]

    elif c2d_idxs:
        vals1 = coord_map[c2d_names[0]].tolist()
        vals2 = coord_map[c2d_names[1]].tolist()
        color2d_grid = _make_color2d_palette(len(vals1), len(vals2))
        c2d_cmap: Dict[Any, Tuple] = {
            (v1, v2): tuple(color2d_grid[i, j].tolist())
            for i, v1 in enumerate(vals1)
            for j, v2 in enumerate(vals2)
        }
        df["_color"] = [
            c2d_cmap[(r[c2d_names[0]], r[c2d_names[1]])]
            for _, r in df[c2d_names].iterrows()
        ]

    else:
        default_c = _rgba(sns.color_palette("husl", 1)[0])
        df["_color"] = [default_c] * len(df)

    # ------------------------------------------------------------------
    # Step 6 — size (linewidth) column
    # ------------------------------------------------------------------
    size_names = [names[i] for i in size_idxs]
    if size_idxs:
        all_vals = [coord_map[n].tolist() for n in size_names]
        combos = list(itertools.product(*all_vals))
        lws = _make_linewidths(len(combos), sizes, scale_linewidth_sqrt)
        lw_map: Dict[Any, float] = dict(zip(combos, lws.tolist()))
        if len(size_idxs) == 1:
            df["_lw"] = df[size_names[0]].map(
                {v: lw_map[(v,)] for v in coord_map[size_names[0]].tolist()}
            )
        else:
            df["_lw"] = [
                lw_map[tuple(r[n] for n in size_names)]
                for _, r in df[size_names].iterrows()
            ]
    else:
        df["_lw"] = (sizes[0] + sizes[1]) / 2.0

    # ------------------------------------------------------------------
    # Step 7 — style (linestyle) column
    # ------------------------------------------------------------------
    if style_idx is not None:
        style_name = names[style_idx]
        style_vals = coord_map[style_name].tolist()
        ls_list = _make_style_linestyles(len(style_vals), dashes)
        ls_map: Dict[Any, Any] = dict(zip(style_vals, ls_list))
        df["_ls"] = df[style_name].map(ls_map)
    else:
        df["_ls"] = "-"

    # ------------------------------------------------------------------
    # Step 8 — facet row/col columns
    # ------------------------------------------------------------------
    row_col = col_col = None
    if row_idx is not None:
        df["_row"] = df[names[row_idx]]
        row_col = "_row"
    if col_idx is not None:
        df["_col"] = df[names[col_idx]]
        col_col = "_col"

    # ------------------------------------------------------------------
    # Step 9 — FacetGrid
    # ------------------------------------------------------------------
    facet_kw: Dict[str, Any] = dict(height=height, aspect=aspect)
    if row_col:
        facet_kw["row"] = row_col
    if col_col:
        facet_kw["col"] = col_col
        if col_wrap is not None and row_col is None:
            facet_kw["col_wrap"] = col_wrap

    g = sns.FacetGrid(df, **facet_kw)

    # Columns that uniquely identify one line within a facet
    line_cols = list(dict.fromkeys(
        hue_names
        + c2d_names
        + ([names[style_idx]] if style_idx is not None else [])
        + size_names
    ))
    if not line_cols:
        df["_line"] = 0
        line_cols = ["_line"]

    # Closure: all outer vars captured by reference
    _t = time_name
    _lc = line_cols
    _kw = kwargs

    def _draw(data: pd.DataFrame, **_ignored: Any) -> None:
        ax = plt.gca()
        for _, grp in data.groupby(_lc, sort=False):
            grp_s = grp.sort_values(_t)
            first = grp_s.iloc[0]
            ax.plot(
                grp_s[_t].values,
                grp_s["value"].values,
                color=first["_color"],
                linewidth=first["_lw"],
                linestyle=first["_ls"],
                alpha=alpha,
                **_kw,
            )

    g.map_dataframe(_draw)

    # ------------------------------------------------------------------
    # Step 10 — labels
    # ------------------------------------------------------------------
    g.set_axis_labels(
        xlabel if xlabel is not None else time_name,
        ylabel if ylabel is not None else "value",
    )
    if row_col or col_col:
        row_tmpl = (f"{names[row_idx]} = {{row_name}}" if row_idx is not None else "")
        col_tmpl = (f"{names[col_idx]} = {{col_name}}" if col_idx is not None else "")
        g.set_titles(row_template=row_tmpl, col_template=col_tmpl)

    if title:
        g.figure.suptitle(title, y=1.02)

    # ------------------------------------------------------------------
    # Step 11 — legends
    # ------------------------------------------------------------------
    if legend:
        legend_ax = g.axes.flat[-1]

        if hue_idxs:
            if len(hue_idxs) == 1:
                cname = hue_names[0]
                vals = coord_map[cname].tolist()
                _patch_legend(
                    legend_ax,
                    labels=[str(v) for v in vals],
                    colors=[hue_cmap[v] for v in vals],  # type: ignore[possibly-undefined]
                    title=cname,
                )
            else:
                combos_leg = list(lin_cmap.keys())  # type: ignore[possibly-undefined]
                labels_leg = [
                    ", ".join(f"{n}={v}" for n, v in zip(hue_names, c))
                    for c in combos_leg
                ]
                _patch_legend(
                    legend_ax,
                    labels=labels_leg,
                    colors=[lin_cmap[c] for c in combos_leg],  # type: ignore
                    title=" × ".join(hue_names),
                )

        elif c2d_idxs and color2d_grid is not None:
            g.figure.subplots_adjust(right=0.88)
            _color2d_legend(
                g.figure,
                color2d_grid,
                ax_names=c2d_names,
                ax_labels=[
                    coord_map[c2d_names[0]].tolist(),
                    coord_map[c2d_names[1]].tolist(),
                ],
            )

        if size_idxs:
            all_vals_leg = [coord_map[n].tolist() for n in size_names]
            combos_sz = list(itertools.product(*all_vals_leg))
            labels_sz = [
                (str(c[0]) if len(c) == 1 else
                 ", ".join(f"{n}={v}" for n, v in zip(size_names, c)))
                for c in combos_sz
            ]
            widths_sz = [lw_map[c] for c in combos_sz]  # type: ignore[possibly-undefined]
            _linewidth_legend(
                legend_ax,
                labels=labels_sz,
                widths=widths_sz,
                title=" × ".join(size_names),
            )

        if style_idx is not None:
            svals = coord_map[names[style_idx]].tolist()
            sls   = _make_style_linestyles(len(svals), dashes)
            _style_legend(
                legend_ax,
                labels=[str(v) for v in svals],
                linestyles=sls,
                title=names[style_idx],
            )

    g.tight_layout()
    return g
