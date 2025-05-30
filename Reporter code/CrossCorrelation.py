import os
import re
import numpy as np
import tifffile
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import rescale
from skimage.filters import threshold_otsu
from scipy.spatial import ConvexHull, Delaunay

# ─── USER PARAMETERS ───────────────────────────────────────────────────────────
REPORTER_DIR = "Reporter"
CELLS_DIR    = "Cells"
CELL_PATTERN = r"filtered_seg_"

# physical voxel sizes (µm)
Z_RES = 0.5
XY_RES = 0.065
Z_SCALE = Z_RES / XY_RES          # factor to upsample z to isotropic

# ─── UTILITY FUNCTIONS ─────────────────────────────────────────────────────────
def load_series(folder, pattern=None):
    files = [f for f in os.listdir(folder)
             if f.lower().endswith(('.tif','.tiff'))
             and (pattern is None or re.search(pattern, f))]
    series = []
    for fn in files:
        m = re.search(r'(\d+)', fn)
        if not m:
            continue
        t = int(m.group(1))
        vol = tifffile.imread(os.path.join(folder, fn))
        series.append((t, vol))
    series.sort(key=lambda x: x[0])
    times, vols = zip(*series)
    return list(times), list(vols)

times,    reporter_vols = load_series(REPORTER_DIR)
_,        cell_labels   = load_series(CELLS_DIR, pattern=CELL_PATTERN)
reporter_iso = [
    rescale(vol,
            (Z_SCALE, 1, 1),
            order=1,               # linear interpolation
            preserve_range=True,
            anti_aliasing=True)
    for vol in reporter_vols
]
cell_iso = [
    (rescale(lbl,
             (Z_SCALE, 1, 1),
             order=0,             # nearest-neighbor
             preserve_range=True) > 0)
    for lbl in cell_labels
]

all_pts = np.vstack([
    np.column_stack(np.nonzero(mask))
    for mask in cell_iso
])
hull = ConvexHull(all_pts)
delaunay = Delaunay(all_pts[hull.vertices])

Z, Y, X = cell_iso[0].shape
grid = np.column_stack(np.nonzero(np.ones((Z, Y, X), dtype=bool)))
in_hull = delaunay.find_simplex(grid) >= 0
hull_mask = in_hull.reshape((Z, Y, X))

binary_reporter = []
for rep in reporter_iso:
    vals = rep[hull_mask]
    thresh = threshold_otsu(vals)
    bin_vol = rep > thresh
    binary_reporter.append(bin_vol)

results = []
for t, rep_bin, mask in zip(times, binary_reporter, cell_iso):
    r = rep_bin[hull_mask].ravel().astype(float)
    m = mask[hull_mask].ravel().astype(float)
    phi = np.corrcoef(r, m)[0,1]
    results.append((t, phi))

df = pd.DataFrame(results, columns=["timepoint", "binary_phi"])
df.to_csv("binary_cross_correlation_timeseries.csv", index=False)
print("Saved: binary_cross_correlation_timeseries.csv")

plt.figure(figsize=(6,4))
plt.plot(df["timepoint"], df["binary_phi"], marker='o', linestyle='-')
plt.xlabel("Timepoint")
plt.ylabel("Cross-correlation")
plt.tight_layout()
plt.show()
