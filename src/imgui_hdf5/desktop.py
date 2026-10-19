"""
`imgui-hdf5 --install-desktop`: an app-menu entry and icon for the current user, so the viewer opens from the
launcher and from a file manager's Open with on .h5 / .hdf5 files.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

NAME = "imgui-hdf5"
MIME = "application/x-hdf;application/x-hdf5;"
ICON = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
<rect x="2" y="2" width="60" height="60" rx="12" fill="#2b2d31"/>
<g stroke="#4a9eff" stroke-width="4" fill="none" stroke-linecap="round">
<path d="M18 16v32M18 26h14M18 44h14"/></g>
<g fill="#4a9eff"><rect x="34" y="20" width="16" height="12" rx="3"/><rect x="34" y="38" width="16" height="12" rx="3"/>
<circle cx="18" cy="14" r="5"/></g>
</svg>
"""


def data_home() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")


def entry(exe: str) -> str:
    return "\n".join([
        "[Desktop Entry]",
        "Type=Application",
        "Name=imgui_hdf5",
        "Comment=View an HDF5 file, read-only",
        f'Exec="{exe}" %f',
        f"Icon={NAME}",
        "Terminal=false",
        "Categories=Science;DataVisualization;",
        f"MimeType={MIME}",
        "",
    ])


def install(exe: str = "") -> list[Path]:
    """Write the entry and icon under $XDG_DATA_HOME; Exec is this executable unless one is given."""
    exe = exe or os.path.abspath(sys.argv[0])
    apps, icons = data_home() / "applications", data_home() / "icons" / "hicolor" / "scalable" / "apps"
    apps.mkdir(parents=True, exist_ok=True)
    icons.mkdir(parents=True, exist_ok=True)
    desktop, icon = apps / f"{NAME}.desktop", icons / f"{NAME}.svg"
    desktop.write_text(entry(exe))
    icon.write_text(ICON)
    if shutil.which("update-desktop-database"):
        subprocess.run(["update-desktop-database", str(apps)], check=False, capture_output=True)
    return [desktop, icon]
