using TiffImages
using Images
using ImageTransformations
using ImageFiltering
using ImageMorphology
using Statistics
using CairoMakie

# Activate PDF backend and set font theme
CairoMakie.activate!(type = "pdf")
set_theme!(fonts=(regular="Helvetica", bold="Helvetica"))

function main()
    # -- User Inputs --
    print("CELL dir: ");         cell_dir      = chomp(readline())
    print("REPORTER dir: ");     rep_dir       = chomp(readline())
    print("downsample ratio XY (e.g. 0.5): ");  ratio_xy   = parse(Float64, chomp(readline()))
    print("downsample ratio Z  (e.g. 0.5): ");  ratio_z    = parse(Float64, chomp(readline()))
    print("cube side length (in downsampled pixels): "); cube_len      = parse(Int,     chomp(readline()))
    print("distance bin interval (in downsampled pixels): "); bin_interval = parse(Float64, chomp(readline()))
    print("Time interval (in minutes): ");   TIME_INTERVAL_MIN = parse(Float64, chomp(readline()))
    print("XY pixel resolution (in microns): ");   PX_SIZE_XY = parse(Float64, chomp(readline()))
    print("Plot filename: ");         filename      = chomp(readline())

    # -- Gather & Align Timepoints --
    cell_files = filter(f -> occursin("filtered_seg_", f) && endswith(f, ".tif"),
                        readdir(cell_dir))
    rep_files  = filter(endswith(".tif"), readdir(rep_dir))
    proc_files = filter(f -> startswith(f, "processed_cells_") && endswith(f, ".tif") && !occursin("cp_masks",f),
                        readdir(cell_dir))

    get_cell_time(fn) = parse(Int, match(r"filtered_seg_(\d+)", fn).captures[1])
    get_rep_time(fn)  = parse(Int, match(r"(\d+)", fn).match) - 1
    get_proc_time(fn) = parse(Int, match(r"processed_cells_(\d+)\.tif", fn).captures[1])

    cell_map = Dict(get_cell_time(f) => f for f in cell_files)
    rep_map  = Dict(get_rep_time(f)  => f for f in rep_files)
    proc_map = Dict(get_proc_time(f) => f for f in proc_files)

    # only keep times with all three channels
    times = sort(collect(intersect(intersect(keys(cell_map), keys(rep_map)),
                                   keys(proc_map))))

    # -- Storage for kymograph means --
    all_means = Vector{Vector{Float64}}(undef, length(times))

    for (i, t) in enumerate(times)
        # Load stacks
        cell_stack = TiffImages.load(joinpath(cell_dir, cell_map[t]))
        rep_stack  = TiffImages.load(joinpath(rep_dir,  rep_map[t]))
        proc_stack = TiffImages.load(joinpath(cell_dir, proc_map[t]))

        # Downsample
        cell_ds = imresize(cell_stack, ratio=(ratio_xy, ratio_xy, ratio_z))
        rep_ds  = imresize(rep_stack,  ratio=(ratio_xy, ratio_xy, ratio_z))
        proc_ds = imresize(proc_stack, ratio=(ratio_xy, ratio_xy, ratio_z))

        # Binarize segmentation to get ROI mask
        blur_buf = cell_ds .> 0.0

        # Largest component → ROI bounding box & centroid
        lbl   = label_components(blur_buf)
        idx   = argmax(component_lengths(lbl))
        cent  = component_centroids(lbl)[idx]
        box   = component_boxes(lbl)[idx]
        y1,y2 = box[1][1], box[2][1]
        x1,x2 = box[1][2], box[2][2]
        z1,z2 = box[1][3], box[2][3]

        # Crop volumes to ROI
        sub_blur = @views blur_buf[y1:y2, x1:x2, z1:z2]
        sub_rep  = @views rep_ds[  y1:y2, x1:x2, z1:z2]
        sub_proc = @views proc_ds[ y1:y2, x1:x2, z1:z2]

        # Mask processed‐cell volume by segmentation
        sub_proc .= sub_proc .* sub_blur

        # Distance‐transform–based kymograph setup
        cent_sub = (
            round(Int, cent[1]) - y1 + 1,
            round(Int, cent[2]) - x1 + 1,
            round(Int, cent[3]) - z1 + 1
        )
        seed = falses(size(sub_blur))
        seed[cent_sub...] = true
        dmap = distance_transform(feature_transform(seed))

        # Bin index per voxel
        bin_idx = Int.(floor.(dmap ./ bin_interval) .+ 1)
        n_bins  = maximum(bin_idx)

		# 1) total voxels per bin
		counts_total = zeros(Int, n_bins)
		for idx in eachindex(bin_idx)
			counts_total[bin_idx[idx]] += 1
		end

		# 2) during your existing loop over `inds`, also count cell‐mask voxels
		sums_rep   = zeros(n_bins)
		sums_proc  = zeros(n_bins)
		counts_blur = zeros(Int, n_bins)
        inds      = findall(sub_blur)
		@inbounds for I in inds
			b = bin_idx[I]
			sums_rep[b]    += sub_rep[I]
			sums_proc[b]   += sub_proc[I]
			counts_blur[b] += 1
		end

		# 3) compute occupancy fraction
		occupancy = counts_blur ./ counts_total

		# 4) compute your signal as before…
		means = zeros(n_bins)
		for b in 1:n_bins
			means[b] = sums_proc[b] > 0 ? sums_rep[b] / sums_proc[b] : 0.0
		end

		# …then zero‐out (or set to missing) any bin with <5% occupancy
		THRESH = 0.01
		means[occupancy .< THRESH] .= 0.0     # or `. = missing` if you prefer

        all_means[i] = means
    end

    # Build matrix (bins × times)
    mat = hcat(all_means...)

    # X‐axis: time in hours
    time_hours = (times .* TIME_INTERVAL_MIN) ./ 60.0

    # Y‐axis: distance in µm
    px_eff   = PX_SIZE_XY / ratio_xy
    bin_idxs = 1:size(mat,1)
    dist_um  = (bin_idxs .- 0.5) .* bin_interval .* px_eff

    # Filter out all‐zero rows
    keep = vec(any(mat .!= 0, dims=2))
    mat_filt     = mat[keep, :]
    dist_um_filt = dist_um[keep]

    # Plot & save heatmap
    fig = Figure(size=(5*72, 3*72))
    ax  = CairoMakie.Axis(fig[1,1])
    hm = heatmap!(ax,
             time_hours,
             dist_um_filt,
             transpose(mat_filt);
             colormap = :lajolla)
	Colorbar(fig[:, end+1], hm, label="Normalized intensity (a.u.)")
    ax.xlabel = "Time (h)"
    ax.ylabel = "Distance from center (µm)"
    CairoMakie.save(filename*".pdf", fig)
end

# Run
main()
