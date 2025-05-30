import tifffile as tif
import napari
from natsort import natsorted
import numpy as np
import os

cells = natsorted([f for f in os.listdir("Cells") if "filtered_seg" in f])
cells_unprocessed = natsorted([f for f in os.listdir("Cells") if "_cp_masks" in f])
matrix = natsorted([f for f in os.listdir("Reporter")])

cell_images = []
cell_images_unprocessed = []
reporter_images = []

for f in cells:
    cell_images.append(tif.imread("Cells/"+f))

for f in cells_unprocessed:
    cell_images_unprocessed.append(tif.imread("Cells/"+f))

for f in matrix:
    reporter_images.append(tif.imread("Reporter/"+f))

cell_images = np.array(cell_images)
cell_images_unprocessed = np.array(cell_images_unprocessed)
reporter_images = np.array(reporter_images)

viewer = napari.view_image(reporter_images)
labels_layer = viewer.add_labels(cell_images, name='segmentation')
labels_layer = viewer.add_labels(cell_images_unprocessed, name='segmentation')
napari.run()
