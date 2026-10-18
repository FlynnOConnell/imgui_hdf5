"""
Fixture files written with h5py: `sample.h5` with one of each kind of node, and `masknmf.hdf5` with the layout
masknmf writes (DemixingResults, a rigid registration and its strategy).
"""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

LINE = np.sin(np.linspace(0, 20, 1000))
IMAGE = np.arange(64 * 48, dtype=np.float32).reshape(64, 48)
MOVIE = np.arange(5 * 3 * 32 * 40, dtype=np.int16).reshape(5, 3, 32, 40)
CSR = np.array([[0, 2, 0, 0, 1], [0, 0, 0, 0, 0], [3, 0, 0, 4, 0], [0, 0, 5, 0, 0]], dtype=np.float32)
BIG_SHAPE = (32, 4096, 4096)  # 2 GiB float32 in memory; only frame BIG_FRAME is written
BIG_FRAME = 3
LONG = 50_000_000


def write_csr(group: h5py.Group, dense: np.ndarray) -> None:
    rows, cols = np.nonzero(dense)
    group["data"] = dense[rows, cols]
    group["indices"] = cols.astype(np.int32)
    group["indptr"] = np.searchsorted(rows, np.arange(dense.shape[0] + 1)).astype(np.int32)


def make_sample(path: Path) -> Path:
    with h5py.File(path.with_name("other.h5"), "w") as o:
        o["data"] = np.arange(3)
    with h5py.File(path, "w") as f:
        f.attrs["title"] = "sample"
        f.attrs["version"] = 2
        f.attrs["created"] = np.bytes_("2026-10-18")
        f.attrs["origin"] = np.array([0.5, 1.5])
        s = f.create_group("scalars")
        s["int"] = 42
        s["float"] = 2.5
        s["bool"] = True
        s["str"] = "hello"
        s["bytes"] = np.bytes_("abcde")
        s["empty"] = h5py.Empty("f4")
        f["strings"] = np.array(["alpha", "beta", "gamma"], dtype=h5py.string_dtype())
        f["labels"] = np.array([b"cell", b"junk"])
        f["line"] = LINE
        f["line"].attrs["units"] = "s"
        f["complex"] = LINE * (1 + 1j)
        f["image"] = IMAGE
        f["movie"] = MOVIE
        f["table"] = np.array([(1, 0.5, b"a"), (2, 1.5, b"bb"), (3, 2.5, b"ccc")],
                              dtype=[("id", "i4"), ("x", "f8"), ("name", "S8")])
        f["zero"] = np.zeros((0, 4))
        f["dtype_t"] = np.dtype("f4")
        links = f.create_group("links")
        links["soft"] = h5py.SoftLink("/line")
        links["broken"] = h5py.SoftLink("/nope")
        links["external"] = h5py.ExternalLink("other.h5", "/data")
        links["hard"] = f["image"]
        loop = f.create_group("loop")
        loop["self"] = loop
        csr = f.create_group("sparse_csr")
        write_csr(csr, CSR)
        csr.attrs["encoding-type"] = "csr_matrix"
        csr.attrs["shape"] = CSR.shape
        csc = f.create_group("sparse_csc")  # 10x layout: shape as a dataset, format only from len(indptr)
        write_csr(csc, CSR.T)
        csc["shape"] = np.array(CSR.shape)
        big = f.create_dataset("big", BIG_SHAPE, dtype="f4", chunks=(1, 512, 512), compression="gzip", shuffle=True)
        big[BIG_FRAME] = np.add.outer(np.arange(4096), np.arange(4096)).astype(np.float32)
        long = f.create_dataset("long", (LONG,), dtype="f4", chunks=(2**20,), compression="gzip")
        long[:2**20] = np.arange(2**20, dtype=np.float32)
    return path


def make_masknmf(path: Path, frames: int = 50, height: int = 16, width: int = 16, k: int = 3, rank: int = 4) -> Path:
    """The groups masknmf.io reads: DemixingResults, RigidRegistrationArray and RigidMotionCorrector."""
    rng = np.random.default_rng(0)
    pixels = height * width

    def strided(group, name, value):
        group[name] = value
        group[name].attrs["layout"] = "strided"

    def coo(group, name, dense):
        g = group.create_group(name)
        rows, cols = np.nonzero(dense)
        strided(g, "indices", np.vstack([rows, cols]).astype(np.int64))
        strided(g, "values", dense[rows, cols].astype(np.float32))
        g["size"] = np.array(dense.shape, dtype=np.int64)
        g.attrs["layout"] = "sparse_coo"

    footprints = np.zeros((pixels, k), dtype=np.float32)
    for j in range(k):
        footprints[rng.choice(pixels, 20, replace=False), j] = rng.random(20)
    with h5py.File(path, "w") as f:
        f.attrs["mbo_provenance"] = json.dumps({"input": "d618db4", "settings": "052e1f6"})
        d = f.create_group("DemixingResults")
        d["shape"] = np.array([frames, height, width], dtype=np.int64)
        strided(d, "mean_image", rng.random((height, width), dtype=np.float32))
        strided(d, "noise_variance_image", rng.random((height, width), dtype=np.float32))
        strided(d, "static_baseline", rng.random(pixels, dtype=np.float32))
        coo(d, "spatial_demixed", footprints)
        strided(d, "temporal_demixed", rng.random((frames, k), dtype=np.float32))
        coo(d, "spatial_compressed", rng.random((pixels, rank)).astype(np.float32))
        strided(d, "temporal_compressed", rng.random((rank, frames), dtype=np.float32))
        r = f.create_group("RigidRegistrationArray")
        strided(r, "shifts", rng.normal(size=(frames, 2)).astype(np.float32))
        r["sinc_margin"] = np.int64(12)
        m = f.create_group("RigidMotionCorrector")
        m["template"] = rng.random((height, width), dtype=np.float32)
        m["max_shifts"] = np.array([15, 15])
        m["batch_size"] = np.int64(300)
    return path


@pytest.fixture(scope="session")
def sample(tmp_path_factory) -> Path:
    return make_sample(tmp_path_factory.mktemp("h5") / "sample.h5")


@pytest.fixture(scope="session")
def masknmf(tmp_path_factory) -> Path:
    return make_masknmf(tmp_path_factory.mktemp("h5") / "masknmf.hdf5")
