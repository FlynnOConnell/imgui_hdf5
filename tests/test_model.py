import h5py
import numpy as np
import pytest
from conftest import BIG_FRAME, BIG_SHAPE, CSR, IMAGE, LINE, LONG, MOVIE

from imgui_hdf5 import model as M


@pytest.fixture
def f(sample):
    with M.open_file(str(sample)) as f:
        yield f


def test_open_is_read_only(f):
    assert f.mode == "r"
    with pytest.raises(Exception):
        f["new"] = 1


def test_children_describe_without_following_links(f):
    nodes = {n.name: n for n in M.children(f, "/links")}
    assert nodes["soft"].kind == "soft" and nodes["soft"].target == "/line"
    assert nodes["broken"].kind == "soft" and nodes["broken"].target == "/nope"
    assert nodes["external"].kind == "external" and nodes["external"].target == "other.h5:/data"
    assert nodes["hard"].kind == "dataset" and nodes["hard"].shape == IMAGE.shape and nodes["hard"].dtype == "float32"
    root = {n.name: n for n in M.children(f)}
    assert root["scalars"].kind == "group" and root["dtype_t"].kind == "datatype"
    assert root["strings"].dtype == "str utf-8 vlen" and root["labels"].dtype == "str ascii [4]"
    assert (root["sparse_csr"].shape, root["sparse_csr"].dtype) == (CSR.shape, "sparse csr float32")
    assert M.node(f, "/").kind == "group" and M.node(f, "/links/soft").kind == "soft"


def test_children_in_natural_order(tmp_path):
    with h5py.File(tmp_path / "n.h5", "w") as f:
        for name in ("cell_10", "cell_2", "Cell_1", "b", "a10b", "a9b"):
            f[name] = 0
    with M.open_file(str(tmp_path / "n.h5")) as f:
        assert [n.name for n in M.children(f)] == ["a9b", "a10b", "b", "Cell_1", "cell_2", "cell_10"]


def test_walk_lists_every_link_once_and_survives_a_cycle(f):
    paths = M.walk(f)
    assert len(paths) == len(set(paths))
    assert {"/links/soft", "/links/external", "/loop/self", "/scalars/str", "/sparse_csr/indptr"} <= set(paths)
    assert not any(p.startswith("/loop/self/") for p in paths)
    assert paths.index("/scalars") < paths.index("/scalars/str")


def test_keep_matches_any_case_with_ancestors(f):
    kept = M.keep(M.walk(f), "STR")
    assert {"/scalars", "/scalars/str", "/strings"} <= kept
    assert "/line" not in kept and "/" not in kept
    assert M.keep(M.walk(f), "zzz") == set()


def test_attributes(f):
    rows = {name: (value, dtype) for name, value, dtype in M.attributes(f)}
    assert rows["title"] == ("sample", "str utf-8 vlen")
    assert rows["version"] == ("2", "int64")
    assert rows["created"][0] == "2026-10-18"
    assert rows["origin"] == ("[0.5, 1.5]", "float64")
    assert M.attributes(f["line"]) == [("units", "s", "str utf-8 vlen")]


def test_info_of_a_chunked_compressed_dataset(f):
    rows = dict(M.info(f["big"]))
    assert rows["chunks"] == "(1, 512, 512)"
    assert rows["compression"] == "shuffle, gzip 4"
    assert rows["size in memory"] == "2.0 GiB"
    assert f["big"].id.get_storage_size() < 50 * 2**20  # only one frame written
    assert dict(M.info(f["image"]))["chunks"] == "contiguous"
    assert dict(M.info(f["image"]))["compression"] == "none"


def test_view_kind(f):
    kinds = {p: M.view_kind(f[p]) for p in (
        "/", "scalars/int", "scalars/str", "scalars/empty", "strings", "labels", "line", "complex", "image", "movie",
        "table", "zero", "sparse_csr", "sparse_csc", "big", "long", "links/hard")}
    assert kinds == {
        "/": "group", "scalars/int": "value", "scalars/str": "value", "scalars/empty": "empty", "strings": "table",
        "labels": "table", "line": "line", "complex": "line", "image": "image", "movie": "stack", "table": "compound",
        "zero": "empty", "sparse_csr": "sparse", "sparse_csc": "sparse", "big": "stack", "long": "line",
        "links/hard": "image",
    }


def test_sparse_layouts_read_as_dense(f):
    csr, csc = M.sparse(f["sparse_csr"]), M.sparse(f["sparse_csc"])
    assert (csr.format, csr.shape, csr.nnz) == ("csr", CSR.shape, 5)
    assert (csc.format, csc.shape, csc.nnz) == ("csc", CSR.shape, 5)
    np.testing.assert_array_equal(M.sparse_dense(f["sparse_csr"], csr), CSR)
    np.testing.assert_array_equal(M.sparse_dense(f["sparse_csc"], csc), CSR)
    assert M.sparse(f["scalars"]) is None


def test_values_and_formatting(f):
    assert M.fmt_value(f["scalars/str"][()]) == "hello"
    assert M.fmt_value(f["scalars/bytes"][()]) == "abcde"
    assert M.fmt_value(f["scalars/empty"][()]) == "(empty)"
    assert M.fmt_value(f["labels"][()]) == "['cell', 'junk']"
    assert M.fmt_shape(None) == "null" and M.fmt_shape(()) == "scalar" and M.fmt_shape((2, 3)) == "2 x 3"
    assert M.fmt_bytes(512) == "512 B" and M.fmt_bytes(3 * 2**20) == "3.0 MiB"
    assert M.fmt_dtype(f["table"].dtype) == "compound (3 fields)"


def test_snippet():
    assert M.snippet("/a/b", "dataset") == 'f["a/b"][...]'
    assert M.snippet("/a", "group") == 'f["a"]'
    assert M.snippet("/", "group") == 'f["/"]'
    assert M.snippet('/q"x', "dataset") == 'f["q\\"x"][...]'


def test_window():
    assert M.window(100, -5, 200) == (0, 100)
    assert M.window(100, 10.5, 20.2) == (10, 22)
    assert M.window(100, 150, 160) == (99, 100)


def test_decimate_keeps_a_one_sample_spike():
    y = np.zeros(100_001)
    y[54_321] = 7.0
    x, d = M.decimate(np.arange(y.size, dtype=float), y, 1000)
    assert d.size <= 1002 and d.max() == 7.0 and x[0] == 0 and x[-1] == y.size - 1
    x, d = M.decimate(np.arange(10.0), np.arange(10.0), 1000)
    assert d.size == 10


def test_read_line(f):
    x, y = M.read_line(f["line"], 100, 200)
    np.testing.assert_array_equal(y, LINE[100:200])
    np.testing.assert_array_equal(x, np.arange(100, 200))
    x, y = M.read_line(f["long"], 0, LONG, max_read=10_000, max_points=1000)
    assert y.size <= 1002 and x[-1] < LONG
    assert y.max() > 2**20 - 2 * LONG / 10_000  # strided read still reaches the end of the written part


def test_read_tile_matches_numpy(f):
    t = M.read_tile(f["image"], (0, 1), (0, 0), (10, 30), (5, 25))
    np.testing.assert_array_equal(t.data, IMAGE[10:30, 5:25])
    assert (t.r0, t.c0, t.r1, t.c1) == (10, 5, 30, 25)
    t = M.read_tile(f["movie"], (2, 3), (4, 1, 0, 0), (0, 32), (0, 40))
    np.testing.assert_array_equal(t.data, MOVIE[4, 1])
    t = M.read_tile(f["movie"], (3, 0), (0, 2, 7, 0), (0, 40), (0, 5))
    np.testing.assert_array_equal(t.data, MOVIE[:, 2, 7, :].T)


def test_read_tile_never_reads_more_than_max_pixels(f):
    t = M.read_tile(f["big"], (1, 2), (BIG_FRAME, 0, 0), (0, BIG_SHAPE[1]), (0, BIG_SHAPE[2]), max_pixels=1024)
    assert t.data.shape == (1024, 1024) and (t.sr, t.sc) == (4, 4)
    assert t.data[3, 5] == 3 * 4 + 5 * 4
    t = M.read_tile(f["big"], (1, 2), (BIG_FRAME, 0, 0), (1001, 3001), (0, 4096), max_pixels=1024)
    assert t.r0 % t.sr == 0 and t.r0 <= 1001 and t.r1 >= 3001
    assert t.data[0, 0] == t.r0


def test_limits():
    a = np.arange(1001.0)
    assert M.limits(a, 0, 100) == (0.0, 1000.0)
    assert M.limits(np.full(4, 3.0)) == (3.0, 4.0)
    assert M.limits(np.array([np.nan, np.inf])) == (0.0, 1.0)


def test_masknmf_layout_uses_the_general_views(masknmf):
    with M.open_file(str(masknmf)) as f:
        d = f["DemixingResults"]
        assert M.view_kind(d) == "group"
        footprints = M.sparse(d["spatial_demixed"])
        assert (footprints.format, footprints.shape) == ("coo", (16 * 16, 3))
        dense = M.sparse_dense(d["spatial_demixed"], footprints)
        assert np.count_nonzero(dense) == footprints.nnz == 60
        assert M.view_kind(d["temporal_demixed"]) == "image"
        assert M.view_kind(d["shape"]) == "line"
        assert M.view_kind(d["mean_image"]) == "image"
        assert M.view_kind(f["RigidRegistrationArray/sinc_margin"]) == "value"
        assert ("layout", "strided", "str utf-8 vlen") in M.attributes(d["mean_image"])
        assert M.attributes(d["spatial_demixed"]) == [("layout", "sparse_coo", "str utf-8 vlen")]
        assert M.attributes(f)[0][0] == "mbo_provenance"
        assert set(M.walk(f)) >= {"/DemixingResults/spatial_demixed/values", "/RigidMotionCorrector/template"}
