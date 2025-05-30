import os
import numpy as np
import pandas as pd
import trackpy as tp
import tifffile
import napari
from natsort import natsorted
from skimage.measure import regionprops_table, label, regionprops
from skimage.filters import threshold_otsu
from scipy.spatial import cKDTree

def filter_entrants(df, threshold_start, otsu_frame):
    """
    Apply a frame-wise step threshold:
      - threshold_start for frames <= otsu_frame
      - 1.0 for frames > otsu_frame
    """
    frames = sorted(df['frame'].unique())
    t_break = otsu_frame

    # initialize with first frame
    t0 = frames[0]
    prev_filtered = df[df['frame'] == t0].copy()
    valid_ids = set(prev_filtered['particle'])
    filtered_frames = [prev_filtered]

    for f in frames[1:]:
        curr = df[df['frame'] == f]
        persistent = curr[curr['particle'].isin(valid_ids)]
        new_pts    = curr[~curr['particle'].isin(valid_ids)]

        # step threshold at the Otsu frame
        thresh_f = threshold_start if f <= t_break else 1.0

        if not prev_filtered.empty and not new_pts.empty:
            tree = cKDTree(prev_filtered[['xum','yum','zum']].values)
            dists, _ = tree.query(new_pts[['xum','yum','zum']].values, k=1)
            keep_new = new_pts.loc[dists <= thresh_f]
        else:
            keep_new = new_pts.iloc[0:0]

        valid_ids.update(keep_new['particle'])
        filtered_current = pd.concat([persistent, keep_new], ignore_index=True)
        filtered_frames.append(filtered_current)
        prev_filtered = filtered_current

    return pd.concat(filtered_frames, ignore_index=True)


def centroids_py(label_stack):
    """
    Compute 3D centroids for each frame of a (T, Z, Y, X) label stack.
    Returns a DataFrame with ['x','y','z','frame','label'] in pixel units.
    """
    records = []
    for t in range(label_stack.shape[0]):
        print(f"Computing centroids for frame {t}")
        vol = label_stack[t]
        props = regionprops_table(vol, properties=('label','centroid'))
        df = pd.DataFrame(props).rename(columns={
            'centroid-2': 'x',
            'centroid-1': 'y',
            'centroid-0': 'z'
        })
        df['frame'] = t
        records.append(df[['x','y','z','frame','label']])
    return pd.concat(records, ignore_index=True)


def main():
    segmentation_folder = "Cells"

    # 1. Gather & sort segmentation mask files
    seg_files = [
        os.path.join(segmentation_folder, f)
        for f in os.listdir(segmentation_folder)
        if "mask" in f
    ]
    seg_files = natsorted(seg_files)

    # 2. Load volumes and stack into (T, Z, Y, X)
    vols = [tifffile.imread(path) for path in seg_files]
    label_stack = np.stack(vols, axis=0)

    # 3. Optional cleanup: keep only largest object in frame 1
    mask1 = label_stack[1] > 0
    labeled1 = label(mask1, connectivity=3)
    if labeled1.max() > 0:
        regions = regionprops(labeled1)
        largest = max(regions, key=lambda r: r.area)
        label_stack[1] *= (labeled1 == largest.label)

    # 4. Compute centroids
    df = centroids_py(label_stack)

    # 5. Convert to microns
    df['xum'] = df['x'] * 0.065
    df['yum'] = df['y'] * 0.065
    df['zum'] = df['z'] * 0.5

    # 6. Link & stub-filter
    print("Starting tracking")
    linked = tp.link_df(
        df,
        search_range=1.0,
        adaptive_stop=0.4,
        pos_columns=['xum','yum','zum']
    )
    filtered = tp.filter_stubs(linked, 3)
    # drop old MultiIndex to avoid ambiguity
    filtered = filtered.reset_index(drop=True)
    print("Finished tracking")

    # 7. Determine Otsu breakpoint on per-frame counts
    counts = filtered.groupby('frame')['particle'].nunique()
    otsu_count = threshold_otsu(counts.values)
    above = counts[counts > otsu_count]
    otsu_frame = int(above.index[0]) if len(above) else counts.index[-1]
    print(f"Otsu count={otsu_count:.1f}, breakpoint at frame {otsu_frame}")

    # 8. Re-filter entrants with a pure time step threshold
    filtered_dynamic = filter_entrants(
        filtered,
        threshold_start=2.0,
        otsu_frame=otsu_frame
    )

    # 9. Save filtered tracks
    out_csv = os.path.join(segmentation_folder, 'filtered_tracks.csv')
    filtered_dynamic.to_csv(out_csv, index=False)
    print(f"Filtered tracks written to {out_csv!r}")

    # 10. Zero-out labels not in filtered tracks and write per-frame TIFFs
    for t in range(label_stack.shape[0]):
        keep = filtered_dynamic.loc[
            filtered_dynamic['frame'] == t, 'label'
        ].unique()
        vol = label_stack[t]
        vol[~np.isin(vol, keep)] = 0
        out_path = os.path.join(segmentation_folder, f"filtered_seg_{t}.tif")
        tifffile.imwrite(
            out_path,
            vol,
            imagej=True,
            metadata={'axes': 'ZYX'}
        )

    # 11. Launch Napari for inspection
    viewer = napari.Viewer()
    viewer.add_labels(
        label_stack,
        name='filtered segmentation',
        metadata={'axes': 'TZYX'}
    )
    napari.run()


if __name__ == "__main__":
    main()

