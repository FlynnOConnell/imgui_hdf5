"""
What the window reads from an HDF5 file, without imgui: the tree, attributes, dataset info, the view a node gets,
and reads of only the part on screen.
"""

import math
import posixpath
import re
from dataclasses import dataclass
from typing import Optional, Sequence

import h5py
import numpy as np

LIMIT_READ = 64 * 2**20  # bytes read without asking first
MAX_POINTS = 4000  # points per line; a longer line is drawn as the min and max of each bin
MAX_LINE_READ = 2**22  # samples read for one line view; a wider view reads every k-th sample
MAX_PIXELS = 512  # samples per image side; a larger view reads every k-th row and column


@dataclass
class Node:
    """One link in the file: a group, a dataset, a committed datatype, or a soft or external link (not followed)."""

    path: str
    kind: str  # "group", "dataset", "datatype", "soft", "external"
    shape: Optional[tuple] = None
    dtype: str = ""
    target: str = ""  # soft: the path; external: "file:path"

    @property
    def name(self) -> str:
        return posixpath.basename(self.path) or "/"

    @property
    def is_link(self) -> bool:
        return self.kind in ("soft", "external")


def open_file(path: str) -> h5py.File:
    return h5py.File(path, "r")


def describe(group: h5py.Group, name: str, path: str) -> Node:
    """The Node for group[name], reading the link itself so soft and external links are not followed."""
    link = group.get(name, getlink=True)
    if isinstance(link, h5py.SoftLink):
        return Node(path, "soft", target=link.path)
    if isinstance(link, h5py.ExternalLink):
        return Node(path, "external", target=f"{link.filename}:{link.path}")
    obj = group[name]
    if isinstance(obj, h5py.Group):
        sp = sparse(obj)
        return Node(path, "group", sp.shape, f"sparse {sp.format} {sp.dtype}") if sp else Node(path, "group")
    if isinstance(obj, h5py.Dataset):
        return Node(path, "dataset", obj.shape, fmt_dtype(obj.dtype))
    return Node(path, "datatype", dtype=fmt_dtype(obj.dtype))


def natural(name: str) -> list:
    """Sort key putting cell_2 before cell_10."""
    return [(0, int(part), "") if part.isdigit() else (1, 0, part.lower()) for part in re.split(r"(\d+)", name)]


def children(f: h5py.File, path: str = "/") -> list[Node]:
    group = f[path]
    return [describe(group, name, posixpath.join(path, name)) for name in sorted(group, key=natural)]


def node(f: h5py.File, path: str) -> Node:
    if path == "/":
        return Node("/", "group")
    parent, name = posixpath.split(path)
    return describe(f[parent], name, path)


def walk(f: h5py.File) -> list[str]:
    """Every path in the file, depth first. Links are listed, not followed; a group hard-linked twice is entered once."""
    paths: list[str] = []
    seen = {f.id}
    stack = [(f, "/")]
    while stack:
        group, path = stack.pop()
        inner = []
        for name in group:
            child = posixpath.join(path, name)
            paths.append(child)
            if isinstance(group.get(name, getlink=True), h5py.HardLink):
                obj = group[name]
                if isinstance(obj, h5py.Group) and obj.id not in seen:
                    seen.add(obj.id)
                    inner.append((obj, child))
        stack.extend(reversed(inner))
    return paths


def keep(paths: Sequence[str], text: str) -> set[str]:
    """The paths containing text (any case), and their ancestors."""
    text = text.lower()
    out: set[str] = set()
    for p in paths:
        if text in p.lower():
            while p != "/" and p not in out:
                out.add(p)
                p = posixpath.dirname(p)
    return out


# ----------------------------------------------------------------------
# formatting
# ----------------------------------------------------------------------


def fmt_dtype(dt: np.dtype) -> str:
    string = h5py.check_string_dtype(dt)
    if string is not None:
        return f"str {string.encoding} " + (f"[{string.length}]" if string.length else "vlen")
    vlen = h5py.check_vlen_dtype(dt)
    if vlen is not None:
        return f"vlen {fmt_dtype(vlen)}"
    ref = h5py.check_ref_dtype(dt)
    if ref is not None:
        return "region reference" if ref is h5py.RegionReference else "object reference"
    if h5py.check_enum_dtype(dt) is not None:
        return f"enum {dt}"
    if dt.names:
        return f"compound ({len(dt.names)} fields)"
    if dt.subdtype is not None:
        base, shape = dt.subdtype
        return f"{fmt_dtype(base)} {shape}"
    if dt.kind == "V":
        return f"opaque [{dt.itemsize}]"
    return str(dt)


def fmt_shape(shape: Optional[tuple]) -> str:
    if shape is None:
        return "null"
    if shape == ():
        return "scalar"
    return " x ".join(str(n) for n in shape)


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return ""


def fmt_value(value) -> str:
    if isinstance(value, h5py.Empty):
        return "(empty)"
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, (float, np.floating)):
        return f"{value:.6g}"
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "SO" and value.size:
            value = np.vectorize(lambda v: v.decode("utf-8", "replace") if isinstance(v, bytes) else v, otypes=[object])(value)
        return np.array2string(value, threshold=20, edgeitems=3, max_line_width=10**6, separator=", ")
    return str(value)


def snippet(path: str, kind: str) -> str:
    """h5py code that reads the node at path from an open file f."""
    key = '"' + (path.lstrip("/") or "/").replace("\\", "\\\\").replace('"', '\\"') + '"'
    return f"f[{key}][...]" if kind == "dataset" else f"f[{key}]"


# ----------------------------------------------------------------------
# attributes and dataset info
# ----------------------------------------------------------------------


def attributes(obj) -> list[tuple[str, str, str]]:
    """(name, value, dtype) per attribute; one that cannot be read says why in its value."""
    rows = []
    for name in obj.attrs:
        try:
            dtype = fmt_dtype(obj.attrs.get_id(name).dtype)
        except Exception:
            dtype = "?"
        try:
            value = fmt_value(obj.attrs[name])
        except Exception as e:
            value = f"(cannot read: {e})"
        rows.append((name, value, dtype))
    return rows


def nbytes(ds: h5py.Dataset) -> int:
    """Bytes in memory when read whole (pointers only, for variable length types)."""
    return int(ds.size or 0) * ds.dtype.itemsize


def filters(ds: h5py.Dataset) -> list[str]:
    plist = ds.id.get_create_plist()
    out = []
    for i in range(plist.get_nfilters()):
        _, _, values, name = plist.get_filter(i)
        name = name.decode("utf-8", "replace") if isinstance(name, bytes) else str(name)
        out.append(f"gzip {values[0]}" if name == "deflate" and values else name)
    return out


def info(ds: h5py.Dataset) -> list[tuple[str, str]]:
    rows = [("shape", str(ds.shape) if ds.shape is not None else "null"), ("dtype", fmt_dtype(ds.dtype))]
    if ds.maxshape is not None and ds.maxshape != ds.shape:
        rows.append(("max shape", "(" + ", ".join("unlimited" if n is None else str(n) for n in ds.maxshape) + ")"))
    rows.append(("chunks", str(ds.chunks) if ds.chunks else "contiguous"))
    rows.append(("compression", ", ".join(filters(ds)) or "none"))
    disk, memory = ds.id.get_storage_size(), nbytes(ds)
    rows.append(("size on disk", fmt_bytes(disk)))
    rows.append(("size in memory", fmt_bytes(memory)))
    if disk and memory:
        rows.append(("memory / disk", f"{memory / disk:.2f}"))
    return rows


# ----------------------------------------------------------------------
# views
# ----------------------------------------------------------------------


@dataclass
class Sparse:
    """A 2-D sparse matrix stored as a group: CSR or CSC (data, indices, indptr) or COO (indices, values, size)."""

    format: str  # "csr", "csc", "coo"
    shape: tuple[int, int]
    nnz: int
    dtype: np.dtype
    parts: tuple[str, ...]

    @property
    def dense_bytes(self) -> int:
        return self.shape[0] * self.shape[1] * self.dtype.itemsize


def _text(value) -> str:
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value[()]
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def _datasets(group: h5py.Group, names) -> Optional[list[h5py.Dataset]]:
    out = [group.get(n) for n in names]
    return out if all(isinstance(d, h5py.Dataset) for d in out) else None


def sparse(group: h5py.Group) -> Optional[Sparse]:
    """The group's sparse layout, or None. The format comes from the group's attributes, else from len(indptr)."""
    try:
        if parts := _datasets(group, ("data", "indices", "indptr")):
            data, indices, indptr = parts
            if data.ndim != 1 or indices.shape != data.shape or indptr.ndim != 1 or data.dtype.kind not in "biufc":
                return None
            if "shape" in group.attrs:
                shape = group.attrs["shape"]
            elif "h5sparse_shape" in group.attrs:
                shape = group.attrs["h5sparse_shape"]
            elif isinstance(group.get("shape"), h5py.Dataset):
                shape = group["shape"][()]
            else:
                return None
            shape = tuple(int(n) for n in shape)
            fmt = ""
            for key in ("encoding-type", "h5sparse_format", "format"):
                if key in group.attrs:
                    fmt = _text(group.attrs[key]).lower()[:3]
            if fmt not in ("csr", "csc"):
                fmt = "csc" if indptr.shape[0] == shape[1] + 1 and indptr.shape[0] != shape[0] + 1 else "csr"
            if len(shape) != 2 or indptr.shape[0] != shape[0 if fmt == "csr" else 1] + 1:
                return None
            return Sparse(fmt, shape, data.shape[0], data.dtype, ("data", "indices", "indptr"))
        if parts := _datasets(group, ("indices", "values", "size")):
            indices, values, size = parts
            if indices.ndim != 2 or indices.shape[0] != 2 or values.shape != (indices.shape[1],) or size.shape != (2,):
                return None
            if values.dtype.kind not in "biufc":
                return None
            shape = tuple(int(n) for n in size[()])
            return Sparse("coo", shape, values.shape[0], values.dtype, ("indices", "values", "size"))
    except (KeyError, TypeError, ValueError, OSError):
        return None
    return None


def sparse_dense(group: h5py.Group, sp: Sparse) -> np.ndarray:
    """The whole matrix as a dense array; duplicate entries add up."""
    out = np.zeros(sp.shape, dtype=sp.dtype)
    if sp.format == "coo":
        rows, cols = group["indices"][()]
        np.add.at(out, (rows, cols), group["values"][()])
        return out
    data, indices, indptr = group["data"][()], group["indices"][()], group["indptr"][()]
    major = np.repeat(np.arange(indptr.shape[0] - 1), np.diff(indptr))
    np.add.at(out, (major, indices) if sp.format == "csr" else (indices, major), data)
    return out


def view_kind(obj) -> str:
    """
    The view for obj: "group", "sparse", "empty", "compound", "value" (a scalar), "table" (strings and other
    non-numbers), "line" (1-D), "image" (2-D) or "stack" (3-D and up).
    """
    if isinstance(obj, h5py.Group):
        return "sparse" if sparse(obj) else "group"
    if not isinstance(obj, h5py.Dataset):
        return "none"
    if obj.shape is None or obj.size == 0:
        return "empty"
    if obj.dtype.names:
        return "compound"
    if obj.shape == ():
        return "value"
    if obj.dtype.kind not in "biufc":
        return "table"
    return {1: "line", 2: "image"}.get(obj.ndim, "stack")


def to_float(a: np.ndarray) -> np.ndarray:
    """float64 for plotting; complex as its magnitude."""
    a = np.asarray(a)
    if a.dtype.kind == "c":
        a = np.abs(a)
    return np.ascontiguousarray(a, dtype=np.float64)


def window(n: int, lo: float, hi: float) -> tuple[int, int]:
    """The index range [i0, i1) covering lo..hi, clipped to [0, n), at least one sample."""
    i0 = min(max(int(math.floor(lo)), 0), n - 1)
    i1 = min(max(int(math.ceil(hi)) + 1, i0 + 1), n)
    return i0, i1


def decimate(x: np.ndarray, y: np.ndarray, max_points: int = MAX_POINTS) -> tuple[np.ndarray, np.ndarray]:
    """Past max_points samples, each bin is drawn as its min then its max, so a spike one sample wide still shows."""
    if y.size <= max_points:
        return x, y
    width = int(math.ceil(y.size / (max_points // 2)))
    starts = np.arange(0, y.size, width)
    out = np.empty(2 * starts.size)
    out[0::2], out[1::2] = np.minimum.reduceat(y, starts), np.maximum.reduceat(y, starts)
    ends = np.minimum(starts + width, y.size) - 1
    return np.column_stack([x[starts], x[ends]]).ravel(), out


def read_line(ds, i0: int, i1: int, max_read: int = MAX_LINE_READ, max_points: int = MAX_POINTS):
    """(x, y) of ds[i0:i1] as float64, reading at most max_read samples, decimated to max_points."""
    step = max(1, math.ceil((i1 - i0) / max_read))
    y = to_float(ds[i0:i1:step])
    x = (i0 + np.arange(y.size) * step).astype(np.float64)
    return decimate(x, y, max_points)


def select(fixed: Sequence[int], picks: dict) -> tuple:
    """An index into an N-D array: the slice in picks on those axes, the position in fixed on the others."""
    return tuple(picks.get(i, fixed[i]) for i in range(len(fixed)))


@dataclass
class Tile:
    """Every sr-th row from r0 and sc-th column from c0 of the image formed by two axes, as float64."""

    data: np.ndarray
    r0: int
    c0: int
    sr: int
    sc: int

    @property
    def r1(self) -> int:
        return self.r0 + self.data.shape[0] * self.sr

    @property
    def c1(self) -> int:
        return self.c0 + self.data.shape[1] * self.sc


def read_tile(arr, axes: tuple[int, int], fixed: Sequence[int], rows: tuple[int, int], cols: tuple[int, int],
              max_pixels: int = MAX_PIXELS) -> Tile:
    """
    The image with axes[0] as rows and axes[1] as columns, the other axes at fixed, over rows and cols, reading at
    most max_pixels per side. Starts are aligned to the stride, so panning reads the same grid.
    """
    (r0, r1), (c0, c1) = rows, cols
    sr, sc = max(1, math.ceil((r1 - r0) / max_pixels)), max(1, math.ceil((c1 - c0) / max_pixels))
    r0, c0 = r0 // sr * sr, c0 // sc * sc
    a = np.asarray(arr[select(fixed, {axes[0]: slice(r0, r1, sr), axes[1]: slice(c0, c1, sc)})])
    if axes[0] > axes[1]:
        a = a.T
    return Tile(to_float(a), r0, c0, sr, sc)


def limits(a: np.ndarray, lo: float = 1.0, hi: float = 99.5) -> tuple[float, float]:
    """Color limits at the lo and hi percentiles of the finite values, never equal."""
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = (float(v) for v in np.percentile(finite, [lo, hi]))
    return (vmin, vmax) if vmax > vmin else (vmin, vmin + 1.0)
