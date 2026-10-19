"""
Draw the window for a few frames per step on hello_imgui's null backend: no window or GPU. Skips when the backend
cannot start here.
"""

import pytest
from imgui_bundle import hello_imgui, imgui, immapp

from imgui_hdf5 import model as M
from imgui_hdf5.app import App, runner_params

FRAMES = 4


def enable_textures() -> None:
    # the null renderer uploads no font atlas; imgui 1.92 asserts in NewFrame unless the backend claims textures
    imgui.get_io().backend_flags |= imgui.BackendFlags_.renderer_has_textures


def run_frames(app: App, steps) -> None:
    """Call app.gui each frame; steps[i](app) runs before frame i * FRAMES. Exits after the last."""
    state = {"frame": 0}

    def frame() -> None:
        i, f = divmod(state["frame"], FRAMES)
        if i >= len(steps):
            hello_imgui.get_runner_params().app_shall_exit = True
            return
        if f == 0:
            steps[i](app)
        app.gui()
        state["frame"] += 1

    params = runner_params(app)
    params.platform_backend_type = hello_imgui.PlatformBackendType.null
    params.renderer_backend_type = hello_imgui.RendererBackendType.null
    params.ini_folder_type = hello_imgui.IniFolderType.temp_folder
    params.callbacks.post_init = enable_textures
    params.callbacks.show_gui = frame
    try:
        immapp.run(params, immapp.AddOnsParams(with_implot=True))
    except Exception as e:
        if "backend" in str(e).lower():
            pytest.skip(f"null backend unavailable: {e}")
        raise


def select(path):
    return lambda a: a.tree.select(path, reveal=True)


def test_every_node_and_view_renders(sample, masknmf, tmp_path, monkeypatch):
    monkeypatch.setenv("IMGUI_HDF5_HOME", str(tmp_path))
    app = App()
    errors = {}
    steps = [lambda a: None, lambda a: a.open(str(sample))]
    with M.open_file(str(sample)) as f:
        paths = ["/", *M.walk(f)]
    steps += [select(p) for p in paths]
    steps += [
        select("/movie"),
        lambda a: setattr(a.page.views["/movie"], "axes", (0, 3)),
        lambda a: a.page.views["/movie"].fixed.__setitem__(1, 2),
        lambda a: setattr(a.page.views["/movie"], "auto", False),
        lambda a: setattr(a.page.views["/movie"], "table", True),
        select("/image"),
        lambda a: setattr(a.page.views["/image"], "table", True),
        select("/big"),
        lambda a: a.page.views["/big"].fixed.__setitem__(0, 3),
        select("/sparse_csr"),
        lambda a: setattr(a.page.views["/sparse_csr"], "table", True),
        lambda a: a.tree.set_filter("str"),
        lambda a: a.tree.set_filter(""),
        lambda a: setattr(a, "rail_width", 5.0),
        lambda a: setattr(a, "rail_width", None),
        lambda a: setattr(a, "show_dialog", True),
        lambda a: errors.update({p: v.errors for p, v in a.page.views.items() if v.errors}),
        lambda a: a.open(str(masknmf)),
        select("/DemixingResults/spatial_demixed"),
        select("/DemixingResults/temporal_demixed"),
        lambda a: a.open(str(tmp_path / "missing.h5")),
        lambda a: a.close(),
    ]
    run_frames(app, steps)
    assert errors == {}
    assert app.file is None and "missing.h5" in app.error
