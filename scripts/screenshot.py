"""
Screenshots for the README: the app's own framebuffer (hello_imgui.final_app_window_screenshot), never the screen.
Writes a demo file, opens it, selects a node, and saves docs/<name>.png. Needs a desktop session.

    python scripts/screenshot.py            # every shot
    python scripts/screenshot.py movie      # one
"""

import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parent.parent
FRAMES = 30
SHOTS = {
    # name: (path to select, view settings)
    "movie": ("/movie", {"fixed": [120, 0, 0]}),
    "line": ("/traces/cell_3", {}),
}


def write_png(path: Path, rgb: np.ndarray) -> None:
    """An 8-bit RGB PNG, without pillow."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def make_demo(path: Path) -> Path:
    """A small imaging session: a movie of cells, their traces and footprints, and acquisition attributes."""
    rng = np.random.default_rng(1)
    frames, h, w, k = 300, 128, 128, 12
    yy, xx = np.mgrid[:h, :w]
    centers = rng.uniform(12, 116, (k, 2))
    footprints = np.exp(-((yy[None] - centers[:, 0, None, None]) ** 2 + (xx[None] - centers[:, 1, None, None]) ** 2) / 18)
    spikes = rng.random((frames, k)) < 0.02
    kernel = np.exp(-np.arange(40) / 8)
    traces = np.array([np.convolve(spikes[:, j], kernel)[:frames] for j in range(k)]).T
    movie = 100 + 400 * np.einsum("tk,khw->thw", traces, footprints) + rng.normal(0, 8, (frames, h, w))
    with h5py.File(path, "w") as f:
        f.attrs["session"] = "2026-10-18 mouse 4 plane 1"
        f.attrs["frame_rate"] = 30.0
        f.attrs["pixel_size_um"] = [1.2, 1.2]
        m = f.create_dataset("movie", data=movie.astype(np.uint16), chunks=(1, h, w), compression="gzip")
        m.attrs["units"] = "photons"
        f["mean_image"] = movie.mean(axis=0).astype(np.float32)
        f["footprints"] = footprints.astype(np.float32)
        t = f.create_group("traces")
        for j in range(k):
            t[f"cell_{j}"] = (traces[:, j] + rng.normal(0, 0.05, frames)).astype(np.float32)
        f["traces"].attrs["kernel_tau_frames"] = 8
        f["latest"] = h5py.SoftLink("/traces/cell_0")
    return path


def shot(name: str) -> None:
    from imgui_bundle import hello_imgui, immapp

    from imgui_hdf5.app import App, runner_params

    select, settings = SHOTS[name]
    demo = make_demo(Path(tempfile.mkdtemp()) / "session.h5")
    app = App(str(demo), scale=1.0)
    app.tree.select(select, reveal=True)
    count = {"n": 0}
    gui = app.gui

    def frame() -> None:
        count["n"] += 1
        view = app.page.views.get(select)
        if view is not None and settings:
            for key, value in settings.items():
                setattr(view, key, value)
        if count["n"] >= FRAMES:
            hello_imgui.get_runner_params().app_shall_exit = True
        gui()

    params = runner_params(app)
    params.app_window_params.window_geometry.size = (1500, 900)
    params.app_window_params.restore_previous_geometry = False
    params.ini_folder_type = hello_imgui.IniFolderType.temp_folder
    params.fps_idling.enable_idling = False
    params.callbacks.show_gui = frame
    immapp.run(params, immapp.AddOnsParams(with_implot=True))
    image = np.asarray(hello_imgui.final_app_window_screenshot())
    if image.size == 0:
        print(f"{name}: empty framebuffer (needs a desktop session)")
        return
    (REPO / "docs").mkdir(exist_ok=True)
    write_png(REPO / "docs" / f"{name}.png", np.ascontiguousarray(image[..., :3]))
    print(f"docs/{name}.png")


if __name__ == "__main__":
    names = sys.argv[1:] or list(SHOTS)
    if len(names) == 1:
        shot(names[0])
    else:
        for n in names:  # one process per shot: a fresh GL context each
            subprocess.run([sys.executable, __file__, n], check=False)
