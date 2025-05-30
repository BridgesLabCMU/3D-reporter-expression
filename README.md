# 3D-reporter-expression
Code for analyzing gene expression reporter data acquired by timelapses of confocal z-stacks

Genertic steps:
0. Save each channel as its own tif
1. ImageJ rolling ball (radius=50) on cells -- save as 4D tif
2. cellpose_preprocessing.jl (change directories if necessary)
3a. conda activate cellpose (if not already activated)
3b. run cellpose command line command (rename folder as necessary)
4. cellpose_postprocessing.py
5. kymograph or cross-correlation or bulk analysis

kymograph_core.jl will create a kymograph where the reference point is the center of mass of the biofilm, forced to z=0. kymograph_com.jl will create a kymograph where the reference point is just the center of mass of the biofilm. Otherwise, the scripts are identical. CrossCorrelation.py computes the cross-correlation between the binarized cells and binarized reporter. Values will vary between -1 (perfect anti-correlation) and 1 (perfect overlap). The value is unitless. local_cell_analysis.py is a work in progress but could be used down the road for doing analysis on regional heterogeneity in new cell divisions or cell dispersal events. view_segmented.py is just a script for viewing 3D images in napari. Doesn't have to just be used for viewing cell segmentations, but I find it useful for that. BulkBiofilmAnalysis.jl will compute the total number of biofilm cells and the total within-biofilm reporter signal over time and display the result as a plot with two y-axes (one for number of cells, the other for total reporter signal).
