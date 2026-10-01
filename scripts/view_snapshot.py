"""Coarse ASCII view of a captured screen snapshot (`_live_now.npy`).

_SNAP_PATH can be overridden as the first CLI argument.
"""
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SNAP_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else _PROJECT_ROOT / "_live_now.npy"
img = np.load(_SNAP_PATH)
H, W = img.shape[:2]
# sample the screen as a coarse ASCII map of brightness/color
from PIL import Image

im = Image.fromarray(img)
im = im.resize((96, 52))
a = np.array(im)
r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
# classify each cell
out = []
for y in range(52):
    row = ""
    for x in range(96):
        rr, gg, bb = int(r[y, x]), int(g[y, x]), int(b[y, x])
        if gg > rr + 20 and gg > bb + 20:
            row += "G"  # green felt
        elif rr > 200 and gg > 200 and bb > 200:
            row += "W"  # white
        elif bb > rr + 15 and bb > gg:
            row += "B"  # blue
        elif rr < 60 and gg < 60 and bb < 60:
            row += "."  # dark
        elif abs(rr - gg) < 20 and abs(gg - bb) < 20 and rr > 120:
            row += "#"  # light gray
        else:
            row += "o"  # other colored
    out.append(row)
print("\n".join(out))
