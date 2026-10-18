"""ImPlot helpers the views share: colors, line specs, plot flags and the colormap names."""

from typing import Optional

from imgui_bundle import imgui, implot

ImVec4 = imgui.ImVec4

ACCENT = ImVec4(0.40, 0.68, 1.00, 1.0)
DIM = ImVec4(0.62, 0.62, 0.65, 1.0)
LINK = ImVec4(0.55, 0.80, 0.55, 1.0)
WARN = ImVec4(1.0, 0.75, 0.3, 1.0)
ERROR = ImVec4(1.0, 0.4, 0.4, 1.0)

SCALE = 1.0  # the UI scale; line weights are in pixels, so they grow with it

LINE_FLAGS = implot.Flags_.no_legend | implot.Flags_.no_title | implot.Flags_.no_box_select
IMAGE_FLAGS = (
    implot.Flags_.no_legend | implot.Flags_.no_title | implot.Flags_.no_box_select | implot.Flags_.no_mouse_text
    | implot.Flags_.no_menus
)


def spec(color: Optional[ImVec4] = None, weight: float = 1.0) -> implot.Spec:
    return implot.Spec(line_color=color, line_weight=weight * SCALE)


def colormaps() -> list[str]:
    return [implot.get_colormap_name(i) for i in range(implot.get_colormap_count())]
