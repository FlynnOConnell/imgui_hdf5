import ctypes

from imgui_hdf5.app import App


def drop(app: App, *items: str) -> None:
    array = (ctypes.c_char_p * len(items))(*(i.encode() for i in items))
    app.on_drop(None, len(items), array)


def test_drop_keeps_only_existing_files(sample, tmp_path, monkeypatch):
    monkeypatch.setenv("IMGUI_HDF5_HOME", str(tmp_path))
    app = App()
    drop(app, "Classifier: some clipboard text", str(sample))
    assert app.dropped == [str(sample)] and app.error == ""
    app.dropped.clear()
    drop(app, "Classifier: some clipboard text")
    assert app.dropped == [] and "did not contain a file path" in app.error
