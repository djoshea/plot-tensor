# pt — Plot Tensor

`pt` is a Python utility for visualizing multi-dimensional arrays as faceted line plots. It maps each dimension of a tensor to a visual channel (colour, line width, linestyle, facet row, facet column) and renders the result using seaborn.

**Supported array types:** `numpy.ndarray`, JAX `Array`, Penzai `NamedArray`, `xarray.DataArray`

```
pip install pt  # or: uv add --editable /path/to/pt
```

---

## Quick start

```python
import numpy as np
import pt

# 1-D array — single line, time axis auto-detected
pt.line(np.random.randn(200).cumsum())

# 2-D array — first axis auto-mapped to colour, last to time
signals = np.random.randn(8, 200).cumsum(axis=-1)
pt.line(signals)
```

---

## `pt.line`

```python
pt.line(tensor, *, time=None, x=None,
        hue=None, color=None, color2d=None,
        style=None, size=None,
        row=None, col=None,
        dim_names=None, coords=None,
        palette=None, sizes=(0.5, 2.5), scale_linewidth_sqrt=False,
        dashes=None, height=3.0, aspect=1.5, col_wrap=None,
        alpha=0.8, legend=True, title=None, xlabel=None, ylabel=None,
        verbose=False, **kwargs) -> sns.FacetGrid
```

Each keyword argument maps one or more tensor axes to a visual channel. All are optional — `pt.line` applies smart defaults for under-specified cases.

---

## Axis specification

### Naming axes

For plain numpy / JAX arrays, axes are named `dim_0`, `dim_1`, … by default. Supply names via `dim_names`:

```python
# List of strings
pt.line(x, dim_names=['batch', 'layer', 'time'])

# List of (name, coordinate_labels) tuples — names and labels together
pt.line(x, dim_names=[
    ('batch', None),                        # labels default to 0, 1, 2, …
    ('layer', ['L0', 'L1', 'L2', 'L3']),
    ('time',  np.linspace(0, 1, T)),
])
```

Penzai `NamedArray` and `xarray.DataArray` supply names (and xarray coordinates) automatically.

### Coordinate labels

Override or supplement labels with the `coords` dict. Keys may be axis names or integer indices. `None` or `()` defaults to `np.arange(n)`.

```python
pt.line(x,
    dim_names=['batch', 'layer', 'time'],
    coords={
        'layer': ['L0', 'L1', 'L2', 'L3'],
        'time':  np.linspace(0.0, 1.0, T),
    })
```

---

## Channel reference

### `time` / `x` — x-axis *(aliases)*

The axis that becomes the x-axis of each line plot. One or neither may be specified.

- **Auto-detection:** if any axis is named `time`, `t`, `T`, or `x`, it is automatically bound without needing `time=`.
- **Fallback:** the last axis.

```python
pt.line(x, time='t')      # explicit
pt.line(x, x='t')         # same thing
pt.line(x)                # auto-detected if a dim is named 'time'
```

### `hue` / `color` — line colour *(aliases)*

Maps one or more axes to line colour. `hue` and `color` are identical; use whichever you prefer.

```python
# Single axis → sequential palette
pt.line(x, hue='layer')

# Multiple axes → Cartesian-product, linearised onto a single palette
pt.line(x, color=['layer', 'head'])
```

**Default palette:** `husl` for ≤ 12 values (perceptually uniform categorical), `viridis` for > 12. Override with `palette=`:

```python
pt.line(x, hue='layer', palette='tab10')
pt.line(x, hue='layer', palette=['#e41a1c', '#377eb8', '#4daf4a'])
```

### `color2d` — 2-D colour palette

Maps **exactly two** axes to a 2-D HLS colour grid: the first axis varies hue across the colour wheel (0.05 → 0.85), the second varies lightness (0.35 → 0.65). This keeps both axes visually distinguishable simultaneously.

`color2d` is mutually exclusive with `hue` / `color`.

```python
# head axis → hue direction, layer axis → lightness direction
pt.line(x, dim_names=['run', 'head', 'layer', 't'],
        color2d=['head', 'layer'], col='run')
```

A swatch-grid legend is placed on the right margin of the figure.

### `style` — linestyle

Maps one axis to linestyle, cycling: solid → dashed → dotted → dash-dot → …

```python
pt.line(x, hue='layer', style='condition')

# Custom dash patterns (matplotlib dash specs)
pt.line(x, style='condition',
        dashes=[(None,None), (4, 2), (1, 1)])
```

### `size` — line width

Maps one or more axes to linewidth, linearly interpolated across `sizes=(min, max)`.

```python
pt.line(x, hue='layer', size='run', sizes=(0.5, 3.0))

# Area-proportional scaling (sqrt mode)
pt.line(x, size='run', sizes=(0.5, 3.0), scale_linewidth_sqrt=True)
```

### `row` / `col` — facet axes

```python
pt.line(x, hue='layer', row='batch', col='condition')

# Single faceting dimension with wrapping
pt.line(x, hue='layer', col='batch', col_wrap=4)
```

---

## Unassigned axes

Any axis not mapped to a channel is **mean-reduced** with a `UserWarning`:

```python
# 'batch' is unassigned → averaged over, warning emitted
pt.line(x, dim_names=['batch', 'layer', 'time'], hue='layer')
# UserWarning: Axes ['batch'] are not assigned to any channel and will be mean-reduced.
```

Pass `verbose=True` to print a table of how every axis is mapped before plotting:

```python
pt.line(x, dim_names=['batch', 'layer', 'time'],
        hue='layer', row='batch', verbose=True)
```

```
dim           shape  role            coords
------------  -----  --------------  ------------------------
batch             4  row             [0, 1, 2, 3]
layer             6  hue             [0 .. 5]  (6)
time            100  x-axis          [0.00 .. 0.99]  (100)
```

---

## Named array types

### xarray DataArray

Dimension names and coordinate values are extracted automatically:

```python
import xarray as xr

da = xr.DataArray(
    data,
    dims=['batch', 'layer', 'time'],
    coords={'layer': ['L0','L1','L2'], 'time': t_values},
)
pt.line(da, hue='layer', row='batch')
```

### Penzai NamedArray

```python
from penzai.core import named_axes as na

arr = na.NamedArray.wrap(data, ('batch', 'layer', 'time'))
pt.line(arr, hue='layer', row='batch')
```

### JAX arrays

Converted to numpy automatically. Pass `dim_names` / `coords` to annotate axes.

```python
import jax.numpy as jnp
pt.line(jnp.array(data), dim_names=['layer', 'time'], hue='layer')
```

---

## Figure and aesthetic options

| Parameter | Default | Description |
|---|---|---|
| `height` | `3.0` | Height of each facet in inches |
| `aspect` | `1.5` | Width-to-height ratio per facet |
| `col_wrap` | `None` | Wrap columns (only when `row` is not used) |
| `alpha` | `0.8` | Line opacity |
| `legend` | `True` | Show colour / size / style legends |
| `title` | `None` | Figure suptitle |
| `xlabel` | `None` | x-axis label (defaults to axis name) |
| `ylabel` | `None` | y-axis label (defaults to `"value"`) |
| `**kwargs` | | Forwarded to `ax.plot()` (e.g. `marker='o'`, `linestyle='--'`) |

---

## Return value

`pt.line` returns a `seaborn.FacetGrid`, giving full access to the underlying figure and axes:

```python
g = pt.line(x, hue='layer', row='batch')

g.set(xlim=(0, 100), ylim=(-5, 5))
g.set_titles(row_template='batch {row_name}')
g.figure.savefig('output.png', dpi=150, bbox_inches='tight')
```

---

## Examples

### Research workflow: compare activations across layers and conditions

```python
# activations: shape (n_layers=12, n_conditions=4, n_tokens=64)
activations = model.get_activations(inputs)  # numpy array

g = pt.line(
    activations,
    dim_names=['layer', 'condition', 'token'],
    coords={
        'layer':     [f'L{i}' for i in range(12)],
        'condition': ['base', 'prefix', 'fewshot', 'finetune'],
    },
    time='token',
    hue='condition',
    row='layer',
    col_wrap=4,
    height=2.0,
    aspect=2.0,
    title='Layer activations by condition',
)
```

### 2-D colour map: heads × layers

```python
# attention: shape (n_heads=8, n_layers=6, seq_len=128)
g = pt.line(
    attention,
    dim_names=['head', 'layer', 'position'],
    color2d=['head', 'layer'],
    time='position',
    alpha=0.6,
    sizes=(0.5, 1.5),
)
```

### xarray with automatic coordinates

```python
import xarray as xr

da = xr.DataArray(
    training_curves,                        # shape (runs, steps)
    dims=['run', 'step'],
    coords={
        'run':  [f'seed={s}' for s in seeds],
        'step': np.arange(n_steps) * log_interval,
    },
)

g = pt.line(da, hue='run', xlabel='Training step', ylabel='Loss')
```

### Combining channels

```python
# signals: (subject=10, condition=3, electrode=64, time=500)
g = pt.line(
    signals,
    dim_names=['subject', 'condition', 'electrode', 'time'],
    time='time',
    hue='condition',
    style='condition',   # redundant encoding: colour + linestyle
    size='electrode',    # thicker lines for higher electrode indices
    row='subject',
    sizes=(0.3, 2.0),
    alpha=0.5,
    verbose=True,
)
```

---

## Installation

```bash
# From PyPI (when published)
pip install pt

# Editable install from local clone
uv add --editable /path/to/pt

# Optional extras
pip install pt[jax]     # JAX support
pip install pt[penzai]  # Penzai NamedArray support
pip install pt[xarray]  # xarray DataArray support
```
