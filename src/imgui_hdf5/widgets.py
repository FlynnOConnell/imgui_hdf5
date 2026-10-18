"""Small imgui widgets the window shares: a (?) with a tooltip, key / value tables, and clipped text with a tooltip."""

from imgui_bundle import imgui

from imgui_hdf5.plotting import DIM

TABLE_FLAGS = imgui.TableFlags_.row_bg | imgui.TableFlags_.borders_inner_h | imgui.TableFlags_.resizable


def help_mark(text: str) -> None:
    """A dim (?) after the last item, on its line, with text as its wrapped tooltip."""
    imgui.same_line(0, imgui.get_font_size() * 0.3)
    imgui.text_disabled("(?)")
    if imgui.is_item_hovered():
        tooltip(text)


def tooltip(text: str) -> None:
    imgui.begin_tooltip()
    imgui.push_text_wrap_pos(imgui.get_font_size() * 40)
    imgui.text_unformatted(text)
    imgui.pop_text_wrap_pos()
    imgui.end_tooltip()


def cell(text: str, limit: int = 200) -> None:
    """text in a table cell, cut at limit characters and its first line; the whole of it in a tooltip."""
    short = text.split("\n", 1)[0]
    short = short if len(short) <= limit else short[:limit] + "..."
    imgui.text_unformatted(short)
    if short != text and imgui.is_item_hovered():
        tooltip(text if len(text) < 20_000 else text[:20_000] + "...")


def key_values(name: str, rows) -> None:
    """Two columns: a dim key and its value."""
    if not rows or not imgui.begin_table(f"##{name}", 2, TABLE_FLAGS & ~imgui.TableFlags_.resizable):
        return
    em = imgui.get_font_size()
    imgui.table_setup_column("key", imgui.TableColumnFlags_.width_fixed, max(imgui.calc_text_size(k).x for k, _ in rows) + em)
    imgui.table_setup_column("value")
    for key, value in rows:
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.text_colored(DIM, key)
        imgui.table_next_column()
        cell(value)
    imgui.end_table()
