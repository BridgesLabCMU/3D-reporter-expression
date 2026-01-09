# Global Hull Kymograph 

using TiffImages
using Images
using ImageTransformations
using ImageMorphology
using Statistics
using CairoMakie
using DelimitedFiles
using Interpolations   

# ---- Plot backend/theme ----
CairoMakie.activate!(type = "pdf")
set_theme!(fonts=(regular="Helvetica", bold="Helvetica"))

# Parse integers from filenames like filtered_seg_###.tif and reporter ###.tif
get_cell_time(fn::AbstractString) = parse(Int, match(r"filtered_seg_(\d+)", fn).captures[1])
get_rep_time(fn::AbstractString)  = parse(Int, match(r"(\d+)", fn).match)

# Code 2's polygon fill
function point_in_poly(pt::Tuple{Int,Int}, poly::Vector{Tuple{Int,Int}})
    x, y = pt
    inside = false
    j = length(poly)
    for i in 1:length(poly)
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi + eps()) + xi)
            inside = !inside
        end
        j = i
    end
    return inside
end

# Build a convex hull mask from a binary 2D image 
function filled_convex_hull(bin::BitMatrix)::BitMatrix
    h = convexhull(bin)
    # If your ImageMorphology returns a filled mask already:
    if eltype(h) <: Bool && size(h) == size(bin)
        return h
    end
    # Otherwise assume boundary indices and fill via ray casting
    hull_pts = [(I[1], I[2]) for I in h]
    mask = falses(size(bin))
    @inbounds for r in CartesianIndices(bin)
        mask[r] = point_in_poly((r.I[1], r.I[2]), hull_pts)
    end
    return mask
end

# 2D XY distance map (in pixels) from the center of a 2D mask
function xy_distance_map(mask2d::BitMatrix)
    inds = findall(mask2d)
    @assert !isempty(inds) "Hull mask is empty."
    ys = Int[i[1] for i in inds]; xs = Int[i[2] for i in inds]
    cy = round(Int, mean(ys)); cx = round(Int, mean(xs))
    seed = falses(size(mask2d)); seed[cy, cx] = true
    distance_transform(feature_transform(seed))
end

# Per-bin sums for a 3D volume within a 3D mask, using a 3D bin-index map
function radial_profile_sums(vol::AbstractArray{<:Real,3}, mask::BitArray{3},
                             bin_idx::Array{Int,3}, n_bins::Int)
    sums   = zeros(Float64, n_bins)
    for I in findall(mask)
        b = bin_idx[I]
        @inbounds sums[b] += vol[I]
    end
    return sums
end

function main()
    # ===== User Inputs =====
    print("CELL dir: ");                cell_dir  = chomp(readline())
    print("REPORTER dir: ");            rep_dir   = chomp(readline())

    # Physical pixel sizes to replicate Code 2's Z-only "isotropic" step
    print("XY pixel size (µm): ");      xy_res = parse(Float64, chomp(readline()))
    print("Z  pixel size (µm): ");      z_res  = parse(Float64, chomp(readline()))
    ez_scale = z_res / xy_res           # Code 2: new_z = size(seg,3) * (z_res/xy_res)

    # Analysis downsampling (applied AFTER hull is built)
    print("Analysis downsample XY (e.g. 0.5): ");  ratio_xy = parse(Float64, chomp(readline()))
    print("Analysis downsample Z  (e.g. 0.5): ");  ratio_z  = parse(Float64, chomp(readline()))

    # Kymograph settings
    print("Distance bin interval (in DOWNsampled pixels): "); bin_interval = parse(Float64, chomp(readline()))
    print("Time interval (in minutes): ");       TIME_INTERVAL_MIN = parse(Float64, chomp(readline()))
    print("Plot filename (no extension): ");      filename = chomp(readline())

    # ---- Gather files / align timepoints ----
    cell_files = filter(f -> occursin("filtered_seg_", f) && endswith(f, ".tif"), readdir(cell_dir))
    rep_files  = filter(endswith(".tif"), readdir(rep_dir))

    cell_map = Dict(get_cell_time(f) => f for f in cell_files)
    rep_map  = Dict(get_rep_time(f)  => f for f in rep_files)
    times = sort(collect(intersect(keys(cell_map), keys(rep_map))))
    @assert !isempty(times) "No overlapping timepoints across CELL and REPORTER."

    # ---- Build GLOBAL 2D HULL----
    global_hull2d_init = false
    global_hull2d = falses(1,1)

    for t in times
        seg = TiffImages.load(joinpath(cell_dir, cell_map[t]))

        # Z-only upsample to isotropic with nearest-neighbor (Code 2)
        new_z = Int(round(size(seg,3) * ez_scale))
        seg_iso = imresize(seg, (size(seg,1), size(seg,2), new_z),
                           method=BSpline(Constant()))  # nearest-neighbor

        # 2D max projection → binarize
        seg2d = mapslices(maximum, seg_iso; dims=3)[:, :, 1]
        bin   = seg2d .> 0

        # Convex hull boundary → filled (Code 2)
        hull2d = filled_convex_hull(bin)

        if !global_hull2d_init
            global_hull2d = copy(hull2d)
            global_hull2d_init = true
        else
            @assert size(global_hull2d) == size(hull2d) "Inconsistent XY sizes across segmentations."
            global_hull2d .|= hull2d
        end
    end
    @assert any(global_hull2d) "Global convex hull is empty."

    # ---- Downsample the GLOBAL hull to analysis resolution (nearest-neighbor) ----
    hull2d_ds = imresize(global_hull2d,
                         (round(Int, size(global_hull2d,1)*ratio_xy),
                          round(Int, size(global_hull2d,2)*ratio_xy));
                         method=BSpline(Constant()))  # NN for masks
    hull2d_ds = hull2d_ds .> 0.5  # back to Bool for distance transform & masking

    # Precompute 2D distance map on the DOWNsampled grid (in pixels)
    dmap2d = xy_distance_map(hull2d_ds)
    maxdist_in_mask = maximum(dmap2d[hull2d_ds .== true])
    n_bins = Int(floor(maxdist_in_mask / bin_interval)) + 1

    # Y-axis in microns (bin centers) on the downsampled grid
    px_eff_xy = xy_res / ratio_xy                     # µm per (downsampled) pixel
    dist_um   = ((1:n_bins) .- 0.5) .* bin_interval .* px_eff_xy

    # Prepare boundary storage (in µm), and target size for per-time footprints
    targetY, targetX = size(hull2d_ds)
    boundary_um = fill(Float64(NaN), length(times))

    # ---- Iterate timepoints: apply same hull to reporter & build kymograph ----
    all_means = Vector{Vector{Float64}}(undef, length(times))
    total_sums = zeros(Float64, length(times))

    for (i, t) in enumerate(times)
        # Reporter: downsample to analysis grid
        rep = TiffImages.load(joinpath(rep_dir, rep_map[t]))
        rep_ds  = imresize(rep, ratio=(ratio_xy, ratio_xy, ratio_z))
        rep_num = Float32.(rep_ds)  # convert Gray{N0f16} to numeric, same shape

        nz = size(rep_num, 3)
        mask3d   = repeat(hull2d_ds, 1, 1, nz)  # extrude global 2D hull through Z
        dmap3d   = repeat(dmap2d,   1, 1, nz)
        bin_idx  = Int.(floor.(dmap3d ./ bin_interval) .+ 1)
        @inbounds bin_idx[bin_idx .> n_bins] .= n_bins  # clamp

        # Total reporter within hull
        total_sums[i] = sum(rep_num[mask3d .== true])

        # Per-bin means
       all_means[i] = radial_profile_sums(rep_num, mask3d, bin_idx, n_bins)


        # --- Per-time biofilm boundary (white line): max radius of current footprint ---
        seg = TiffImages.load(joinpath(cell_dir, cell_map[t]))
        new_z  = Int(round(size(seg,3) * ez_scale))
        seg_iso = imresize(seg, (size(seg,1), size(seg,2), new_z),
                           method=BSpline(Constant()))      # NN, matches Code 2
        seg2d  = mapslices(maximum, seg_iso; dims=3)[:, :, 1]
        bin2d  = seg2d .> 0
        bin_ds = imresize(bin2d, (targetY, targetX), method=BSpline(Constant()))
        bin_ds = bin_ds .> 0.5

        if any(bin_ds)
            # strict option to keep boundary inside global hull:
            # boundary_pix = maximum(dmap2d[(bin_ds .& hull2d_ds) .== true])
            boundary_pix = maximum(dmap2d[bin_ds .== true])
            boundary_um[i] = boundary_pix * px_eff_xy
        end
    end

    # ---- Assemble matrix and axes ----
    mat = hcat(all_means...)  # (bins × times)
    time_hours = (times .* TIME_INTERVAL_MIN) ./ 60.0

    # ---- Save totals CSV ----
    open(filename * "_totals.csv", "w") do io
        writedlm(io, ["time_h" "total_reporter"])
        writedlm(io, hcat(time_hours, total_sums), ',')
    end

    # ---- Plot heatmap + boundary line ----
    fig = Figure(size=(5*72, 3*72))
    ax  = CairoMakie.Axis(fig[1,1])
    hm = heatmap!(ax,
                  time_hours,
                  dist_um,                 # y in microns
                  transpose(mat);          # value = mat[row(bin), col(time)]
                  colormap = :lajolla)
    lines!(ax, time_hours, boundary_um; color=:white, linewidth=2)  # overlay boundary
    Colorbar(fig[:, end+1], hm, label = "Mean reporter (a.u.)")
    ax.xlabel = "Time (h)"
    ax.ylabel = "Distance from global core (µm)"
    CairoMakie.save(filename * ".pdf", fig)

    println("Done. Wrote $(filename).pdf and $(filename)_totals.csv")
end

main()
