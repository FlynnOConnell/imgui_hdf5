"""
The window `imgui-hdf5` opens: the file's tree in a rail on the left, the selected node's page on the right. Files
open read-only, from the command line, or the File menu (imgui_data_loader's dialog).
With imgui_debugger installed (the debug extra), a Debug menu opens a live inspector on the window's state.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from imgui_bundle import hello_imgui, imgui, immapp, implot
from imgui_data_loader import FileDialog, FileDialogConfig

from imgui_hdf5 import __version__, config
from imgui_hdf5 import model as M
from imgui_hdf5 import plotting as P
from imgui_hdf5.tree import Tree
from imgui_hdf5.views import Page
from imgui_hdf5.widgets import help_mark

try:
    from imgui_debugger import DebugTools
except ImportError:  # optional: pip install imgui_hdf5[debug]
    DebugTools = None

TITLE_WINDOW = "imgui_hdf5"
SIZE_WINDOW = (1500, 1000)
WIDTH_RAIL_EM = 30
IMPLOT_PADDINGS = (
    "plot_padding", "label_padding", "legend_padding", "legend_inner_padding", "legend_spacing",
    "mouse_pos_padding", "annotation_padding", "major_tick_len", "minor_tick_len",
)


class App:
    """The open file, its tree and page, the open dialog, the UI scale and the debug panels."""

    def __init__(self, path: str = "", scale: float = 1.0):
        self.path = ""
        self.file = None
        self.tree: Optional[Tree] = None
        self.page = Page()
        self.error = ""
        self.show_dialog = False
        self.typed = ""
        self.dialog = FileDialog(FileDialogConfig(
            title="Open an HDF5 file",
            subtitle="Files open read-only.",
            buttons=[],
            body_draw=self.draw_dialog_body,
            show_options_button=False,
            show_quit_button=False,
            quit_on_escape=False,
            close_on_select=False,
        ))
        self.scale = config.clamp(scale)
        self.scale_applied = 1.0
        self.scale_edit: Optional[float] = None
        self.tools = DebugTools.default(target=self, title="Viewer state") if DebugTools else None
        if path:
            self.open(path)

    def open(self, path: str) -> None:
        try:
            f = M.open_file(path)
        except Exception as e:
            self.error = f"Cannot open {path}: {e}"
            return
        self.close()
        self.path, self.file, self.tree, self.page, self.error = path, f, Tree(f), Page(), ""
        config.add_recent(path)

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
        self.path, self.file, self.tree, self.page = "", None, None, Page()

    # ------------------------------------------------------------------
    # scale
    # ------------------------------------------------------------------

    def apply_scale(self) -> None:
        """Font, imgui and ImPlot sizes to self.scale. Called every frame; does work only when the scale moved."""
        if self.scale == self.scale_applied:
            return
        ratio = self.scale / self.scale_applied
        style = imgui.get_style()
        style.font_scale_main = self.scale
        style.scale_all_sizes(ratio)
        ps = implot.get_style()
        for name in IMPLOT_PADDINGS:
            v = getattr(ps, name)
            setattr(ps, name, imgui.ImVec2(v.x * ratio, v.y * ratio))
        ps.plot_min_size = imgui.ImVec2(ps.plot_min_size.x * ratio, ps.plot_min_size.y * ratio)
        P.SCALE = self.scale
        self.scale_applied = self.scale

    def set_scale(self, scale: float) -> None:
        self.scale = config.clamp(scale)
        config.save({"ui_scale": self.scale})

    def scale_keys(self) -> None:
        """Ctrl + / Ctrl - / Ctrl 0, as in a browser."""
        if not imgui.get_io().key_ctrl:
            return
        if imgui.is_key_pressed(imgui.Key.equal) or imgui.is_key_pressed(imgui.Key.keypad_add):
            self.set_scale(self.scale * 1.1)
        elif imgui.is_key_pressed(imgui.Key.minus) or imgui.is_key_pressed(imgui.Key.keypad_subtract):
            self.set_scale(self.scale / 1.1)
        elif imgui.is_key_pressed(imgui.Key._0) or imgui.is_key_pressed(imgui.Key.keypad0):
            self.set_scale(config.detect_scale())

    # ------------------------------------------------------------------
    # frame
    # ------------------------------------------------------------------

    def gui(self) -> None:
        self.scale_keys()
        self.apply_scale()
        em = imgui.get_font_size()
        rail = min(max(0.3 * imgui.get_content_region_avail().x, 20 * em), WIDTH_RAIL_EM * em)
        if imgui.begin_child("rail", imgui.ImVec2(rail, 0), imgui.ChildFlags_.borders | imgui.ChildFlags_.resize_x):
            self.draw_rail()
        imgui.end_child()
        imgui.same_line()
        if imgui.begin_child("page", imgui.ImVec2(0, 0)):
            if self.error:
                imgui.text_colored(P.ERROR, self.error)
            if self.tree is not None:
                self.page.draw(self.tree)
            else:
                self.dialog.render()
        imgui.end_child()
        if self.show_dialog:
            imgui.set_next_window_size(imgui.ImVec2(26 * em, 22 * em), imgui.Cond_.first_use_ever)
            _, self.show_dialog = imgui.begin("Open file", True)
            self.dialog.render()
            imgui.end()
        if self.tools:
            self.tools.render()

    def draw_menus(self) -> None:
        if imgui.begin_menu("File"):
            if imgui.menu_item("Open...", "", False)[0]:
                self.show_dialog = True
            if imgui.begin_menu("Open recent"):
                recent = config.recent()
                for path in recent:
                    if imgui.menu_item(path, "", False)[0]:
                        self.open(path)
                if not recent:
                    imgui.text_disabled("none")
                imgui.end_menu()
            if imgui.menu_item("Close", "", False, self.file is not None)[0]:
                self.close()
            imgui.separator()
            if imgui.menu_item("Quit", "", False)[0]:
                hello_imgui.get_runner_params().app_shall_exit = True
            imgui.end_menu()
        if self.tools:
            self.tools.draw_menu()

    def draw_dialog_body(self) -> None:
        imgui.separator_text("Path")
        imgui.set_next_item_width(-5 * imgui.get_font_size())
        entered, self.typed = imgui.input_text_with_hint(
            "##typed", "/path/to/file.h5", self.typed, imgui.InputTextFlags_.enter_returns_true,
        )
        imgui.same_line()
        if (imgui.button("Open") or entered) and self.typed.strip():
            self.open(str(Path(self.typed.strip()).expanduser()))
            if self.file is not None:
                self.typed, self.show_dialog = "", False
        imgui.separator_text("Recent files")
        recent = config.recent()
        for i, path in enumerate(recent):
            if imgui.selectable(f"{Path(path).name}##recent{i}", False)[0]:
                self.open(path)
                self.show_dialog = self.show_dialog and self.file is None
            imgui.same_line()
            imgui.text_disabled(str(Path(path).parent))
        if not recent:
            imgui.text_disabled("none")

    def draw_rail(self) -> None:
        imgui.text_colored(P.ACCENT, TITLE_WINDOW)
        imgui.same_line()
        imgui.text_disabled(f"v{__version__}")
        if self.tree is not None:
            imgui.text(Path(self.path).name)
            if imgui.is_item_hovered():
                imgui.set_tooltip(str(Path(self.path).resolve()))
            imgui.same_line()
            imgui.text_disabled("read-only")
        else:
            imgui.text_disabled("No file open.")
        imgui.spacing()
        if imgui.begin_child("tree", imgui.ImVec2(0, -imgui.get_frame_height_with_spacing() * 1.5)):
            if self.tree is not None:
                self.tree.draw()
        imgui.end_child()
        imgui.separator()
        imgui.set_next_item_width(8 * imgui.get_font_size())
        # applied on release: rescaling mid-drag would move the slider out from under the mouse
        shown = self.scale_edit if self.scale_edit is not None else self.scale
        changed, value = imgui.slider_float("text size", shown, config.SCALE_MIN, config.SCALE_MAX, "%.2fx")
        if changed:
            self.scale_edit = value
        if self.scale_edit is not None and not imgui.is_item_active():
            self.set_scale(self.scale_edit)
            self.scale_edit = None
        help_mark("Also Ctrl + / Ctrl -. Ctrl 0 resets to the detected scale. Saved in ~/.imgui_hdf5/settings.json.")


def runner_params(app: App) -> hello_imgui.RunnerParams:
    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = TITLE_WINDOW
    params.app_window_params.window_geometry.size = tuple(int(v * min(app.scale, 1.5)) for v in SIZE_WINDOW)
    params.imgui_window_params.default_imgui_window_type = hello_imgui.DefaultImGuiWindowType.provide_full_screen_window
    params.imgui_window_params.show_menu_bar = True
    params.imgui_window_params.show_menu_app = False
    params.imgui_window_params.show_menu_view = False
    params.imgui_window_params.tweaked_theme.theme = hello_imgui.ImGuiTheme_.darcula_darker
    params.fps_idling.fps_idle = 20
    params.ini_filename = "imgui_hdf5.ini"
    params.ini_folder_type = hello_imgui.IniFolderType.app_user_config_folder
    params.callbacks.show_gui = app.gui
    params.callbacks.show_menus = app.draw_menus
    params.callbacks.default_icon_font = hello_imgui.DefaultIconFont.font_awesome6
    return params


def run(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="imgui-hdf5", description="View an HDF5 file, read-only.")
    parser.add_argument("path", nargs="?", default="", help=".h5 / .hdf5 file to open")
    parser.add_argument("--scale", type=float, default=0.0, help="UI scale, e.g. 2 on a 2x HiDPI panel (default: saved, then detected)")
    args = parser.parse_args(argv)
    app = App(path=args.path, scale=config.ui_scale(args.scale))
    if args.path and app.file is None:
        print(app.error, file=sys.stderr)
    immapp.run(runner_params(app), immapp.AddOnsParams(with_implot=True))
    app.close()


if __name__ == "__main__":
    run()
