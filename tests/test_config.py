from imgui_hdf5 import config


def test_scale_order_override_env_saved_detected(tmp_path, monkeypatch):
    monkeypatch.setenv("IMGUI_HDF5_HOME", str(tmp_path))
    monkeypatch.delenv("IMGUI_HDF5_SCALE", raising=False)
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
    monkeypatch.setenv("GDK_SCALE", "2")
    assert config.ui_scale() == 2.0
    config.save({"ui_scale": 1.25})
    assert config.ui_scale() == 1.25
    monkeypatch.setenv("IMGUI_HDF5_SCALE", "3")
    assert config.ui_scale() == 3.0
    assert config.ui_scale(1.5) == 1.5
    assert config.ui_scale(99) == config.SCALE_MAX


def test_recent_newest_first_without_duplicates(tmp_path, monkeypatch):
    monkeypatch.setenv("IMGUI_HDF5_HOME", str(tmp_path))
    for name in ("a.h5", "b.h5", "a.h5"):
        config.add_recent(str(tmp_path / name))
    assert config.recent() == [str(tmp_path / "a.h5"), str(tmp_path / "b.h5")]
    for i in range(20):
        config.add_recent(str(tmp_path / f"{i}.h5"))
    assert len(config.recent()) == config.MAX_RECENT
    config.save({"ui_scale": 2})
    assert config.recent()[0] == str(tmp_path / "19.h5")
