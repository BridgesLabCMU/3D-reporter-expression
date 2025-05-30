import os, re, glob
import numpy as np
import pandas as pd
import tifffile
import napari

# 1) load tracks & build a lookup of final frame per particle
csv_path = 'Cells/filtered_tracks.csv'
tracks = pd.read_csv(csv_path)
last_frame = tracks.groupby('particle')['frame'].max().to_dict()

# 2) find all your TIFFs & sort by timepoint
cells_dir = 'Cells'
paths = glob.glob(os.path.join(cells_dir, '*filtered_seg_*.tif'))
tp_paths = []
for p in paths:
    m = re.search(r'filtered_seg_(\d+)', os.path.basename(p))
    if m:
        tp_paths.append((int(m.group(1)), p))
tp_paths.sort(key=lambda x: x[0])

# 3) first pass: build **remapped_volumes** and record the set of particles present each frame
remapped_volumes = []
particle_sets     = []
for tp, path in tp_paths:
    seg = tifffile.imread(path)             # (z,y,x)
    sub = tracks[tracks['frame'] == tp]
    mapping = dict(zip(sub['label'], sub['particle']))

    remap = np.zeros_like(seg, dtype=np.int32)
    for lab, pid in mapping.items():
        remap[seg == lab] = int(pid)

    remapped_volumes.append(remap)
    particle_sets.append(set(mapping.values()))

# 4) second pass: carve out “new” and “ended” volumes
seen = set()
new_volumes = []
end_volumes = []

for i, (tp, _) in enumerate(tp_paths):
    remap = remapped_volumes[i]
    present = particle_sets[i]

    # --- brand-new cells at t ---
    new_ids = present - seen
    seen |= present
    mask_new = np.isin(remap, list(new_ids))
    new_volumes.append(np.where(mask_new, remap, 0))

    # --- cells **ending** at t: last_frame == t-1 ---
    dying_ids = {pid for pid, lf in last_frame.items() if lf == tp - 1}
    if i == 0:
        # no “previous” volume at the very first frame
        end_volumes.append(np.zeros_like(remap, dtype=np.int32))
    else:
        prev = remapped_volumes[i - 1]
        mask_end = np.isin(prev, list(dying_ids))
        end_volumes.append(np.where(mask_end, prev, 0))

# 5) stack into 4D
stack_new = np.stack(new_volumes, axis=0)
stack_end = np.stack(end_volumes, axis=0)

# 6) visualize in Napari
viewer = napari.Viewer()
viewer.add_labels(stack_new, name='new_cells',   opacity=0.8, scale=[1]*4)
viewer.add_labels(stack_end, name='ended_cells', opacity=0.8, scale=[1]*4)
napari.run()

