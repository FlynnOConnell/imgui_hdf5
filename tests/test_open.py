import h5py

from imgui_hdf5.app import App


def test_open_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("IMGUI_HDF5_HOME", str(tmp_path / "home"))
    folder = tmp_path / "dir with space"
    folder.mkdir()
    with h5py.File(folder / "a b.h5", "w") as f:
        f["x"] = [1, 2, 3]
    (tmp_path / "notes.txt").write_text("not hdf5")
    monkeypatch.chdir(folder)

    app = App("a b.h5")  # relative, as from the command line
    assert app.file is not None and app.path == str(folder / "a b.h5") and app.error == ""
    app.close()
    for path, reason in [("missing.h5", "no such file"), (str(tmp_path), "a directory"), (str(tmp_path / "notes.txt"), "not an HDF5 file")]:
        app = App(path)
        assert app.file is None and app.error.endswith(reason)
