# Architecture

## Modules

- `app.py`: `App`, the window. The rail on the left holds the file name, the tree and the text size slider; the page
  on the right is `Page.draw`, or imgui_data_loader's `FileDialog` (a path field and recent files in its body slot, no picker buttons) when no file
  is open. File > Open shows the same dialog in a window. The menu bar has File and, when imgui_debugger is
  installed (the `debug` extra), its `DebugTools`. `run(argv)` is the `imgui-hdf5` entry point.
- `model.py`: everything read from the file, no imgui. `Node` per link (`children`, `node`, `walk`, `keep` for the
  filter), `attributes`, `info`, `view_kind`, `sparse` / `sparse_dense`, and the reads of the part on screen:
  `read_line` (strided read, `decimate` to min / max per bin) and `read_tile` (two axes of an N-D array, every k-th
  row and column, other axes fixed).
- `tree.py`: `Tree`, the open file's tree: children read when a group first opens, the selection, the filter, and
  `reveal` (open a path's ancestors and scroll to it).
- `views.py`: `Page` and a `View` per path visited (axes, slider positions, colormap, contrast, cached reads). One
  function per view kind.
- `plotting.py`: colors, `spec`, plot flags, colormap names.
- `widgets.py`: `help_mark`, `tooltip`, `cell` (clipped text with the rest in a tooltip), `key_values`.
- `config.py`: `~/.imgui_hdf5/settings.json`: the UI scale and recent files.

The hard dependencies are imgui-bundle, numpy, h5py and imgui_data_loader. imgui_debugger is optional, imported in a
try in `app.py`. Layout, scale handling and tests follow imgui_ophys.

## Reads

Every read goes through `View.read(name, key, fn)`: `fn` runs when the key changes, and the result (or the error,
shown above the view) is kept. A still frame reads nothing. While a plot's limits move, `View.kept` draws the last
tile or line and reads once they settle, so panning a 2 GB dataset does not stall on every frame. Tables read the
rows the list clipper shows. The only whole reads are a scalar's value and a sparse matrix made dense; the latter
asks first past `LIMIT_READ` (64 MiB).

## Gotchas

- ImPlot's heatmap draws its first row at the top of the bounds. The image's y axis is inverted so row 0 sits at the
  top and the mouse position is the row, so the tile is passed flipped.
- With `Flags_.equal`, ImPlot keeps the x range and crops y to the aspect. `views.image` shapes the plot's box like
  the image so the first fit shows all of it.
- `plot_heatmap` draws a rect per cell, so a tile is at most `MAX_PIXELS` (512) a side.
- A strided read of a chunked, compressed dataset still decompresses every chunk it touches; an overview of a large
  compressed 2-D dataset is slow once, then cached.
- The list clipper calls `step()` more than once a frame with different ranges; each step has its own cache entry.
- No drag and drop: imgui-bundle's GLFW is X11 only, and under XWayland a drop can deliver the clipboard's text
  instead of the file's path.
- `group.get(name, getlink=True)` reads a link without following it; `group[name]` follows soft links. `walk`
  enters only hard links and tracks object ids, so a group linked into itself is entered once.
- h5py reads variable-length strings as bytes; `fmt_value` decodes them.
- Headless tests run on hello_imgui's null backend, which needs `renderer_has_textures` set in `post_init`.
- Screenshots (`scripts/screenshot.py`) read the app's framebuffer with `hello_imgui.final_app_window_screenshot`,
  one process per shot.
