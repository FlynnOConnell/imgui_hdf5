"""
The right panel for the selected node: its path and copy buttons, attributes, dataset info, and a view chosen from
the shape. Reads are of the part on screen and are kept until the view moves, so a still frame reads nothing.
"""

import math
import posixpath
from dataclasses import dataclass, field
from typing import Callable, Optional

import h5py
import numpy as np
from imgui_bundle import imgui, implot

from imgui_hdf5 import model as M
from imgui_hdf5 import plotting as P
from imgui_hdf5.tree import Tree
from imgui_hdf5.widgets import TABLE_FLAGS, cell, help_mark, key_values, tooltip

TABLE_COLUMNS = 64  # columns a 2-D table shows at once
SCROLL_FLAGS = TABLE_FLAGS | imgui.TableFlags_.scroll_y | imgui.TableFlags_.scroll_x | imgui.TableFlags_.borders_outer


@dataclass
class View:
    """What one node's view shows, kept per path so going back to a node finds it as it was."""

    axes: tuple[int, int] = (0, 1)  # the image's rows and columns, or a table's rows and columns
    fixed: list[int] = field(default_factory=list)  # the position on every other axis
    table: bool = False  # a 2-D view as a table instead of a heatmap
    col0: int = 0  # a table's first column
    cmap: int = int(implot.Colormap_.viridis)
    auto: bool = True
    pct: tuple[float, float] = (1.0, 99.5)
    vmin: float = 0.0
    vmax: float = 1.0
    lim: Optional[tuple[float, float, float, float]] = None  # last frame's plot limits, x0 x1 y0 y1
    moving: bool = False  # the limits changed last frame: draw what was read until they settle
    confirmed: bool = False  # the user agreed to a read over LIMIT_READ
    cache: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)

    def read(self, name: str, key, fn: Callable):
        """fn() the first time with this key, then the kept result. A failed read gives None and keeps its error."""
        old = self.cache.get(name)
        if old is not None and old[0] == key:
            return old[1]
        try:
            value = fn()
            self.errors.pop(name, None)
        except Exception as e:
            value = None
            self.errors[name] = f"{type(e).__name__}: {e}"
        self.cache[name] = (key, value)
        return value

    def kept(self, name: str, key, fn: Callable):
        """read(), but while the plot moves, the last result whatever its key."""
        if self.moving and name in self.cache:
            return self.cache[name][1]
        return self.read(name, key, fn)

    def set_lim(self, lim: tuple[float, float, float, float]) -> None:
        self.moving = self.lim is not None and lim != self.lim
        self.lim = lim


def default_view(shape: Optional[tuple]) -> View:
    n = len(shape or ())
    return View(axes=(max(n - 2, 0), max(n - 1, 0)), fixed=[0] * n)


class Page:
    """The selected node's page, with a View per path visited."""

    def __init__(self):
        self.views: dict[str, View] = {}

    def view(self, path: str, shape: Optional[tuple]) -> View:
        if path not in self.views:
            self.views[path] = default_view(shape)
        return self.views[path]

    def draw(self, tree: Tree) -> None:
        try:
            n = tree.node(tree.selected)
        except Exception as e:
            imgui.text_colored(P.ERROR, f"Cannot read {tree.selected}: {e}")
            return
        header(n)
        imgui.push_id(n.path)
        if n.kind == "soft":
            soft_link(tree, n)
        elif n.kind == "external":
            imgui.text(f"External link to {n.target.split(':', 1)[1]} in {n.target.split(':', 1)[0]}.")
            imgui.text_disabled("External files are not opened.")
        else:
            obj = tree.f[n.path]
            shape = obj.shape if isinstance(obj, h5py.Dataset) else None
            v = self.view(n.path, shape)
            attributes(v, obj)
            if isinstance(obj, h5py.Dataset) and imgui.collapsing_header("Dataset info", imgui.TreeNodeFlags_.default_open):
                key_values("info", v.read("info", None, lambda: M.info(obj)) or [])
            imgui.separator_text("Data")
            if v.errors:
                imgui.text_colored(P.ERROR, "Cannot read: " + "; ".join(sorted(set(v.errors.values()))))
            if imgui.begin_child("##view"):
                data(v, obj)
            imgui.end_child()
        imgui.pop_id()


def header(n: M.Node) -> None:
    imgui.text_colored(P.ACCENT, n.path)
    imgui.same_line()
    what = {"soft": "soft link", "external": "external link"}.get(n.kind, n.kind)
    if n.kind == "dataset":
        what = f"dataset  {M.fmt_shape(n.shape)}  {n.dtype}"
    elif n.dtype:
        what = f"{what}  {M.fmt_shape(n.shape) + '  ' if n.shape else ''}{n.dtype}"
    imgui.text_disabled(f"  {what}")
    if imgui.button("Copy path"):
        imgui.set_clipboard_text(n.path)
    imgui.same_line()
    code = M.snippet(n.path, n.kind)
    if imgui.button("Copy h5py snippet"):
        imgui.set_clipboard_text(code)
    if imgui.is_item_hovered():
        tooltip(code)
    imgui.separator()


def soft_link(tree: Tree, n: M.Node) -> None:
    imgui.text(f"Soft link to {n.target}")
    target = posixpath.normpath(posixpath.join(posixpath.dirname(n.path), n.target))
    try:
        exists = tree.f.get(target) is not None
    except Exception:
        exists = False
    if not exists:
        imgui.text_colored(P.WARN, "The target is not in this file.")
    elif imgui.button("Go to target"):
        tree.select(target, reveal=True)


def attributes(v: View, obj) -> None:
    rows = v.read("attrs", None, lambda: M.attributes(obj)) or []
    if not imgui.collapsing_header(f"Attributes ({len(rows)})###attrs", imgui.TreeNodeFlags_.default_open):
        return
    if not rows:
        imgui.text_disabled("none")
        return
    em = imgui.get_font_size()
    height = (min(len(rows), 8) + 1) * imgui.get_text_line_height_with_spacing() + 0.6 * em
    if imgui.begin_table("##attrs", 3, SCROLL_FLAGS, imgui.ImVec2(0, height)):
        imgui.table_setup_scroll_freeze(0, 1)
        imgui.table_setup_column("name", imgui.TableColumnFlags_.width_fixed, 10 * em)
        imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch)
        imgui.table_setup_column("dtype", imgui.TableColumnFlags_.width_fixed, 7 * em)
        imgui.table_headers_row()
        for name, value, dtype in rows:
            imgui.table_next_row()
            imgui.table_next_column()
            cell(name)
            imgui.table_next_column()
            cell(value)
            imgui.table_next_column()
            imgui.text_disabled(dtype)
        imgui.end_table()


# ----------------------------------------------------------------------
# views by kind
# ----------------------------------------------------------------------


def data(v: View, obj) -> None:
    kind = M.view_kind(obj)
    if kind == "group":
        imgui.text_disabled(f"Group with {len(obj)} members.")
    elif kind == "sparse":
        sparse(v, obj)
    elif kind == "empty":
        imgui.text_disabled(f"No values (shape {M.fmt_shape(obj.shape)}).")
    elif kind == "value":
        value(v, obj)
    elif kind == "line":
        line(v, obj)
    elif kind in ("image", "stack"):
        if obj.ndim > 2:
            axis_controls(v, obj.shape)
        image_or_table(v, obj, obj.shape)
    elif kind == "table":
        if obj.ndim == 1:
            column_table(v, obj, "value")
        else:
            axis_controls(v, obj.shape)
            block_table(v, obj, obj.shape)
    elif kind == "compound":
        compound(v, obj)
    else:
        key_values("dtype", [("dtype", M.fmt_dtype(obj.dtype))])


def value(v: View, ds: h5py.Dataset) -> None:
    text = v.read("value", None, lambda: M.fmt_value(ds[()]))
    if text is None:
        return
    lines = min(text.count("\n") + 1, 20)
    size = imgui.ImVec2(-1, lines * imgui.get_text_line_height() + 2 * imgui.get_style().frame_padding.y + 2)
    imgui.input_text_multiline("##value", text, size, imgui.InputTextFlags_.read_only)


def line(v: View, ds: h5py.Dataset) -> None:
    n = ds.shape[0]
    x0, x1 = v.lim[:2] if v.lim else (0, n - 1)
    i0, i1 = M.window(n, x0, x1)
    xy = v.kept("line", (i0, i1), lambda: M.read_line(ds, i0, i1))
    em = imgui.get_font_size()
    height = max(0.55 * imgui.get_content_region_avail().y, 10 * em)
    if implot.begin_plot("##line", imgui.ImVec2(-1, height), P.LINE_FLAGS):
        implot.setup_axes("index", "", 0, implot.AxisFlags_.auto_fit)
        implot.setup_axis_limits(implot.ImAxis_.x1, 0, max(n - 1, 1), imgui.Cond_.once)
        if xy is not None:
            implot.plot_line("##values", xy[0], xy[1], P.spec(P.ACCENT))
        lim = implot.get_plot_limits()
        v.set_lim((lim.x.min, lim.x.max, 0.0, 0.0))
        implot.end_plot()
    if xy is not None:
        note = f"{i1 - i0} samples in view, {xy[1].size} points drawn"
        if xy[1].size < i1 - i0:
            note += " (min and max of each bin)"
        if ds.dtype.kind == "c":
            note += "; complex values plotted as magnitude"
        imgui.text_disabled(note)
    column_table(v, ds, "value")


def axis_controls(v: View, shape: tuple) -> None:
    """Which axes form the image (or table), and a slider for each of the others."""
    em = imgui.get_font_size()
    names = [f"axis {i} ({n})" for i, n in enumerate(shape)]
    for which, label in enumerate(("rows", "columns")):
        imgui.set_next_item_width(9 * em)
        if imgui.begin_combo(label, names[v.axes[which]]):
            for i, name in enumerate(names):
                if imgui.selectable(name, i == v.axes[which])[0] and i != v.axes[which]:
                    # picking the other one's axis swaps the two
                    v.axes = v.axes[::-1] if i == v.axes[1 - which] else tuple(i if k == which else a for k, a in enumerate(v.axes))
                    v.lim = None
            imgui.end_combo()
        imgui.same_line()
    imgui.new_line()
    sliders(v, shape, v.axes)


def sliders(v: View, shape: tuple, shown) -> None:
    """A slider for the position on each axis not in shown."""
    em = imgui.get_font_size()
    for i, n in enumerate(shape):
        if i in shown:
            continue
        imgui.set_next_item_width(20 * em)
        changed, value = imgui.slider_int(f"axis {i}##fixed{i}", v.fixed[i], 0, n - 1)
        if changed:
            v.fixed[i] = value
    if len(shape) > len(shown):
        help_mark("Ctrl+click a slider to type a position.")


def image_or_table(v: View, arr, shape: tuple) -> None:
    if imgui.radio_button("heatmap", not v.table):
        v.table = False
    imgui.same_line()
    if imgui.radio_button("table", v.table):
        v.table = True
    if v.table:
        block_table(v, arr, shape)
    else:
        imgui.same_line(0, 2 * imgui.get_font_size())
        color_controls(v)
        image(v, arr, shape)


def color_controls(v: View) -> None:
    em = imgui.get_font_size()
    names = P.colormaps()
    imgui.set_next_item_width(8 * em)
    if imgui.begin_combo("colormap", names[v.cmap] if v.cmap < len(names) else ""):
        for i, name in enumerate(names):
            if imgui.selectable(name, i == v.cmap)[0]:
                v.cmap = i
        imgui.end_combo()
    imgui.same_line()
    _, v.auto = imgui.checkbox("auto", v.auto)
    imgui.same_line()
    imgui.set_next_item_width(14 * em)
    if v.auto:
        changed, lo, hi = imgui.drag_float_range2("percentiles", v.pct[0], v.pct[1], 0.1, 0.0, 100.0, "%.1f")
        if changed:
            v.pct = (lo, hi)
    else:
        speed = max((v.vmax - v.vmin) / 300, 1e-9)
        changed, lo, hi = imgui.drag_float_range2("limits", v.vmin, v.vmax, speed, 0.0, 0.0, "%.4g")
        if changed:
            v.vmin, v.vmax = lo, max(hi, lo + 1e-12)
    help_mark("auto: the color limits are these percentiles of the values on screen. Off: set the limits by hand.")


def image(v: View, arr, shape: tuple) -> None:
    axes, fixed = v.axes, v.fixed
    h, w = shape[axes[0]], shape[axes[1]]
    x0, x1, y0, y1 = v.lim or (0, w, 0, h)
    rows, cols = M.window(h, y0, y1), M.window(w, x0, x1)
    tile = v.kept("tile", (axes, tuple(fixed), rows, cols), lambda: M.read_tile(arr, axes, fixed, rows, cols))
    if tile is not None and v.auto:
        v.vmin, v.vmax = v.read("limits", (id(tile), v.pct), lambda: M.limits(tile.data, *v.pct))
    em = imgui.get_font_size()
    bar = 5 * em
    avail = imgui.get_content_region_avail()
    size = imgui.ImVec2(max(avail.x - bar - imgui.get_style().item_spacing.x, 10 * em), max(avail.y, 10 * em))
    equal = max(h, w) <= 8 * min(h, w)
    if equal:
        # the plot's box shaped like the image, so the whole image fits at the first fit
        pad = 3.5 * em  # axis ticks and labels
        size.x = min(size.x, (size.y - pad) * w / h + pad)
        size.y = min(size.y, (size.x - pad) * h / w + pad)
    flags = P.IMAGE_FLAGS | (implot.Flags_.equal if equal else 0)
    implot.push_colormap(v.cmap)
    if implot.begin_plot(f"##image{axes}", size, flags):
        implot.setup_axes(f"axis {axes[1]}", f"axis {axes[0]}", 0, implot.AxisFlags_.invert)
        implot.setup_axes_limits(0, w, 0, h, imgui.Cond_.once)
        if tile is not None:
            # heatmap rows run from the top of the bounds down; the y axis is inverted so row 0 is at the top
            implot.plot_heatmap("##data", np.ascontiguousarray(tile.data[::-1]), v.vmin, v.vmax, "",
                                implot.Point(tile.c0, tile.r0), implot.Point(tile.c1, tile.r1))
        if implot.is_plot_hovered():
            p = implot.get_plot_mouse_pos()
            r, c = int(math.floor(p.y)), int(math.floor(p.x))
            if 0 <= r < h and 0 <= c < w:
                index = M.select(fixed, {axes[0]: r, axes[1]: c})
                found = v.read("hover", index, lambda: arr[index])
                text = f"row {r}, col {c}\nvalue {M.fmt_value(found)}"
                if len(shape) > 2:
                    text += f"\nindex {index}"
                tooltip(text)
        lim = implot.get_plot_limits()
        v.set_lim((lim.x.min, lim.x.max, min(lim.y.min, lim.y.max), max(lim.y.min, lim.y.max)))
        implot.end_plot()
    implot.pop_colormap()
    imgui.same_line()
    implot.colormap_scale("##scale", v.vmin, v.vmax, imgui.ImVec2(bar, size.y), "%g", 0, v.cmap)


def rows_table(v: View, name: str, n: int, headers: list[str], key, read: Callable, cells: Callable) -> None:
    """
    n rows with an index column and headers; read(a, b) gives rows a..b, read only as the clipper reaches them, and
    cells(block, i) the texts of row i of the block.
    """
    em = imgui.get_font_size()
    if not imgui.begin_table(f"##{name}", len(headers) + 1, SCROLL_FLAGS, imgui.ImVec2(0, 0)):
        return
    imgui.table_setup_scroll_freeze(1, 1)
    imgui.table_setup_column("index", imgui.TableColumnFlags_.width_fixed, max(4, len(str(n))) * 0.6 * em + em)
    for h in headers:
        imgui.table_setup_column(h, imgui.TableColumnFlags_.width_fixed, 7 * em)
    imgui.table_headers_row()
    clipper = imgui.ListClipper()
    clipper.begin(n)
    step = 0
    while clipper.step():
        a, b = clipper.display_start, clipper.display_end
        block = v.read(f"{name}{step}", (key, a, b), lambda: read(a, b))
        step += 1
        for i in range(a, b):
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text_disabled(str(i))
            if block is None:
                continue
            for text in cells(block, i - a):
                imgui.table_next_column()
                cell(text)
    imgui.end_table()


def column_table(v: View, ds, header: str) -> None:
    """A 1-D dataset as one column."""
    rows_table(v, "column", ds.shape[0], [header], None, lambda a, b: ds[a:b], lambda block, i: [M.fmt_value(block[i])])


def block_table(v: View, arr, shape: tuple) -> None:
    """The 2-D slice of v.axes at v.fixed, TABLE_COLUMNS columns at a time."""
    ra, ca = v.axes
    ncols = shape[ca]
    if ncols > TABLE_COLUMNS:
        imgui.set_next_item_width(20 * imgui.get_font_size())
        _, v.col0 = imgui.slider_int("first column", v.col0, 0, ncols - TABLE_COLUMNS)
    c0 = min(v.col0, max(ncols - TABLE_COLUMNS, 0))
    c1 = min(c0 + TABLE_COLUMNS, ncols)

    def read(a, b):
        block = np.asarray(arr[M.select(v.fixed, {ra: slice(a, b), ca: slice(c0, c1)})])
        return block.T if ra > ca else block

    rows_table(v, "block", shape[ra], [str(c) for c in range(c0, c1)], (v.axes, tuple(v.fixed), c0), read,
               lambda block, i: [M.fmt_value(x) for x in block[i]])


def compound(v: View, ds: h5py.Dataset) -> None:
    """One column per field, over the last axis; a slider for each axis before it."""
    names = list(ds.dtype.names)
    if ds.ndim == 0:
        rows_table(v, "fields", 1, names, None, lambda a, b: np.asarray(ds[()]).reshape(1),
                   lambda block, i: [M.fmt_value(block[i][f]) for f in names])
        return
    axis = ds.ndim - 1
    sliders(v, ds.shape, (axis,))
    rows_table(v, "fields", ds.shape[axis], names, tuple(v.fixed), lambda a, b: ds[M.select(v.fixed, {axis: slice(a, b)})],
               lambda block, i: [M.fmt_value(block[i][f]) for f in names])


def sparse(v: View, group: h5py.Group) -> None:
    sp = v.read("sparse", None, lambda: M.sparse(group))
    if sp is None:
        return
    if not v.fixed:
        v.axes, v.fixed = (0, 1), [0, 0]
    total = sp.shape[0] * sp.shape[1]
    key_values("sparse", [
        ("format", sp.format.upper()),
        ("shape", str(sp.shape)),
        ("stored values", str(sp.nnz)),
        ("density", f"{sp.nnz / total:.3g}" if total else "-"),
        ("dtype", str(sp.dtype)),
        ("members", ", ".join(sp.parts)),
        ("dense size", M.fmt_bytes(sp.dense_bytes)),
    ])
    if sp.dense_bytes > M.LIMIT_READ and not v.confirmed:
        imgui.text_colored(P.WARN, f"As a dense array this takes {M.fmt_bytes(sp.dense_bytes)} of memory.")
        if imgui.button(f"Read as dense ({M.fmt_bytes(sp.dense_bytes)})"):
            v.confirmed = True
        return
    dense = v.read("dense", None, lambda: M.sparse_dense(group, sp))
    if dense is not None:
        image_or_table(v, dense, dense.shape)
