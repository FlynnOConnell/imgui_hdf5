"""
The left panel's tree: one row per link with its shape and dtype, children read when a group is first opened, and a
filter on the path. Soft and external links are rows of their own and never expanded.
"""

from typing import Optional

import h5py
from imgui_bundle import imgui

from imgui_hdf5 import model as M
from imgui_hdf5 import plotting as P
from imgui_hdf5.widgets import tooltip

TREE_FLAGS = (
    imgui.TableFlags_.row_bg | imgui.TableFlags_.borders_v | imgui.TableFlags_.scroll_x | imgui.TableFlags_.scroll_y
    | imgui.TableFlags_.no_borders_in_body
)
NODE_FLAGS = (
    imgui.TreeNodeFlags_.span_all_columns | imgui.TreeNodeFlags_.open_on_arrow | imgui.TreeNodeFlags_.open_on_double_click
    | imgui.TreeNodeFlags_.nav_left_jumps_to_parent
)
LEAF_FLAGS = imgui.TreeNodeFlags_.leaf | imgui.TreeNodeFlags_.no_tree_push_on_open


class Tree:
    """The open file's tree, the selected path and the filter."""

    def __init__(self, f: h5py.File):
        self.f = f
        self.selected = "/"
        self.filter = ""
        self.reveal: Optional[str] = None  # a path whose ancestors open and which scrolls into view on the next frame
        self.cache: dict[str, list[M.Node]] = {}
        self.nodes: dict[str, M.Node] = {}
        self.paths: Optional[list[str]] = None  # every path, walked on the first filter
        self.kept: Optional[set[str]] = None
        self.width = 0.0  # what the drawn rows need, measured each frame, for sizing the rail
        self.widths = [0.0, 0.0, 0.0]
        self.x0 = 0.0

    def children(self, path: str) -> list[M.Node]:
        if path not in self.cache:
            try:
                self.cache[path] = M.children(self.f, path)
            except Exception:
                self.cache[path] = []
            self.nodes.update((n.path, n) for n in self.cache[path])
        return self.cache[path]

    def node(self, path: str) -> M.Node:
        if path not in self.nodes:
            self.nodes[path] = M.node(self.f, path)
        return self.nodes[path]

    def select(self, path: str, reveal: bool = False) -> None:
        self.selected = path
        if reveal:
            self.reveal = path

    def set_filter(self, text: str) -> None:
        self.filter = text
        if not text:
            self.kept = None
            return
        if self.paths is None:
            self.paths = M.walk(self.f)
        self.kept = M.keep(self.paths, text)

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------

    def draw(self) -> None:
        imgui.set_next_item_width(-1)
        changed, text = imgui.input_text_with_hint("##filter", "filter by path", self.filter)
        if changed:
            self.set_filter(text)
        if self.kept is not None:
            imgui.text_disabled(f"{len(self.kept)} matching" if self.kept else "no match")
        if not imgui.begin_table("##tree", 3, TREE_FLAGS, imgui.ImVec2(0, 0)):
            return
        imgui.table_setup_scroll_freeze(0, 1)
        # columns fit their contents every frame, so names are never cut; a narrow rail scrolls sideways instead
        imgui.table_setup_column("name", imgui.TableColumnFlags_.no_hide | imgui.TableColumnFlags_.width_fixed)
        imgui.table_setup_column("shape", imgui.TableColumnFlags_.width_fixed)
        imgui.table_setup_column("dtype", imgui.TableColumnFlags_.width_fixed)
        imgui.table_headers_row()
        # filtered rows open their own state, so clearing the filter leaves the tree as it was
        imgui.push_id("filtered" if self.kept is not None else "tree")
        self.widths = [0.0, 0.0, 0.0]
        self.row(self.node("/"))
        imgui.pop_id()
        imgui.end_table()
        style = imgui.get_style()
        self.width = sum(self.widths) + 6 * style.cell_padding.x + style.scrollbar_size

    def row(self, n: M.Node) -> None:
        imgui.table_next_row()
        imgui.table_next_column()
        if n.path == "/":
            self.x0 = imgui.get_cursor_screen_pos().x
        name_x = imgui.get_cursor_screen_pos().x - self.x0 + imgui.get_font_size() + 2 * imgui.get_style().frame_padding.x
        self.measure(0, name_x + imgui.calc_text_size(n.name).x)
        group = n.kind == "group"
        flags = NODE_FLAGS | (0 if group else LEAF_FLAGS)
        if n.path == self.selected:
            flags |= imgui.TreeNodeFlags_.selected
        if group and (self.kept is not None or (self.reveal or "").startswith(n.path.rstrip("/") + "/")):
            imgui.set_next_item_open(True)
        elif n.path == "/":
            imgui.set_next_item_open(True, imgui.Cond_.once)
        if n.is_link:
            imgui.push_style_color(imgui.Col_.text, P.LINK)
        opened = imgui.tree_node_ex(f"{n.name}##{n.path}", flags)
        if n.is_link:
            imgui.pop_style_color()
        if imgui.is_item_clicked() and not imgui.is_item_toggled_open():
            self.select(n.path)
        if imgui.is_item_focused() and imgui.get_io().nav_visible and n.path != self.selected:
            self.select(n.path)  # arrow keys move the selection
        if n.path == self.reveal:
            imgui.set_scroll_here_y()
            self.reveal = None
        imgui.table_next_column()
        if n.is_link:
            self.text(1, f"-> {n.target}")
            if imgui.is_item_hovered():
                tooltip(n.target)
        elif n.shape is not None or n.kind == "dataset":
            self.text(1, M.fmt_shape(n.shape))
        imgui.table_next_column()
        self.text(2, {"soft": "soft link", "external": "external link"}.get(n.kind, n.dtype))
        if opened and group:
            for child in self.children(n.path):
                if self.kept is None or child.path in self.kept:
                    self.row(child)
            imgui.tree_pop()

    def measure(self, column: int, width: float) -> None:
        self.widths[column] = max(self.widths[column], width)

    def text(self, column: int, text: str) -> None:
        self.measure(column, imgui.calc_text_size(text).x)
        imgui.text_disabled(text)
