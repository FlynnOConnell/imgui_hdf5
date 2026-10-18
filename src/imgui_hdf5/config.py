"""
The window's own settings on disk (`~/.imgui_hdf5/settings.json`, or under `IMGUI_HDF5_HOME`): the UI scale and the
recent files.

GLFW reports a content scale of 1 on some HiDPI setups (Hyprland through XWayland, for one), so hello_imgui never
learns the monitor is scaled and draws 15 px text on a 2x panel. The window keeps its own scale instead: the saved
one, else `IMGUI_HDF5_SCALE`, else a guess from `GDK_SCALE` or Hyprland's monitor scale.
"""

import json
import os
import subprocess
from pathlib import Path

SCALE_MIN, SCALE_MAX = 0.75, 4.0
MAX_RECENT = 10


def home() -> Path:
    return Path(os.environ.get("IMGUI_HDF5_HOME", Path.home() / ".imgui_hdf5"))


def load() -> dict:
    try:
        return json.loads((home() / "settings.json").read_text())
    except (OSError, ValueError):
        return {}


def save(values: dict) -> None:
    try:
        home().mkdir(parents=True, exist_ok=True)
        (home() / "settings.json").write_text(json.dumps({**load(), **values}, indent=1))
    except OSError:
        pass  # a read-only home only costs the remembered settings


def recent() -> list[str]:
    """Recently opened files, newest first."""
    paths = load().get("recent", [])
    return [p for p in paths if isinstance(p, str)] if isinstance(paths, list) else []


def add_recent(path: str) -> None:
    path = str(Path(path).resolve())
    save({"recent": [path, *(p for p in recent() if p != path)][:MAX_RECENT]})


def clamp(scale: float) -> float:
    return round(min(max(float(scale), SCALE_MIN), SCALE_MAX), 2)


def detect_scale() -> float:
    """The desktop's scale as far as the environment tells: GDK_SCALE, then Hyprland's largest monitor scale."""
    try:
        if float(os.environ.get("GDK_SCALE", "0")) > 0:
            return float(os.environ["GDK_SCALE"])
    except ValueError:
        pass
    if os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"):
        try:
            out = subprocess.run(["hyprctl", "monitors", "-j"], capture_output=True, text=True, timeout=2).stdout
            return max(float(m.get("scale", 1)) for m in json.loads(out))
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return 1.0


def ui_scale(override: float = 0.0) -> float:
    """The scale to start with: override, the env var, the saved one, then the detected one."""
    if override > 0:
        return clamp(override)
    try:
        if float(os.environ.get("IMGUI_HDF5_SCALE", "0")) > 0:
            return clamp(float(os.environ["IMGUI_HDF5_SCALE"]))
    except ValueError:
        pass
    saved = load().get("ui_scale")
    if isinstance(saved, (int, float)) and saved > 0:
        return clamp(saved)
    return clamp(detect_scale())
