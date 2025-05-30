using TiffImages
using Images 
using Interpolations    
using ImageMorphology  
using Makie
using CairoMakie
using Statistics
using NaturalSort

# Physical resolutions (μm)
xy_res = 0.065
z_res  = 0.5
ez_scale = z_res / xy_res

# Directories
cells_dir    = "Cells"
reporter_dir = "Reporter"
cells_files    = sort(filter(f -> occursin("filtered_seg", f), readdir(cells_dir)), lt=natural)
reporter_files = sort(readdir(reporter_dir), lt=natural)

n = length(cells_files)
cell_counts   = zeros(Int, n)
hull_masks    = Vector{BitMatrix}(undef, n)
reporter_sums = zeros(Float64, n)

function point_in_poly(pt::Tuple{Int,Int}, poly::Vector{Tuple{Int,Int}})
    x, y = pt
    inside = false
    j = length(poly)
    for i in 1:length(poly)
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
            inside = !inside
        end
        j = i
    end
    return inside
end

for (i, fname) in enumerate(cells_files)
    segstack = TiffImages.load(joinpath(cells_dir, fname))

    new_z = Int(round(size(segstack,3) * ez_scale))
    seg_iso = imresize(segstack,
                       (size(segstack,1), size(segstack,2), new_z),
                       method=BSpline(Constant()))

    seg2d = mapslices(maximum, seg_iso; dims=3)[:,:,1]

    cell_counts[i] = length(unique(segstack)) - 1 

    bin = seg2d .> 0
    hull_idx = convexhull(bin)
    hull_pts = [(I[1], I[2]) for I in hull_idx]
    mask = falses(size(bin))
    for r in CartesianIndices(bin)
        mask[r] = point_in_poly((r.I[1], r.I[2]), hull_pts)
    end
    hull_masks[i] = mask
end

global_hull = reduce((a,b) -> a .| b, hull_masks)

for (i, fname) in enumerate(reporter_files)
    repstack = TiffImages.load(joinpath(reporter_dir, fname))
    new_z = Int(round(size(repstack,3) * ez_scale))
    rep_iso = imresize(repstack,
                       (size(repstack,1), size(repstack,2), new_z),
                       method=BSpline(Constant()))

    total = 0.0
    for z in 1:size(rep_iso,3)
        slice = rep_iso[:,:,z]
        total += sum( slice[global_hull] )
    end
    reporter_sums[i] = total
end

times = 0:20:(n-1)*20
f = Figure(size=(4.5*72, 3*72))

ax1 = CairoMakie.Axis(f[1, 1], yticklabelcolor = "#FC8D62")
ax2 = CairoMakie.Axis(f[1, 1], yticklabelcolor = "#8DA0CB", yaxisposition = :right)

lines!(ax1, times, cell_counts, color = "#FC8D62", linewidth=2)
lines!(ax2, times, reporter_sums, color = "#8DA0CB", linewidth=2) 
ax1.xlabel = "Time (min)" 
ax1.ylabel = "Number of cells" 
ax2.ylabel = "RbmA-3xF" 
ax1.rightspinecolor = "#8DA0CB"
ax1.leftspinecolor = "#FC8D62"
CairoMakie.save("plot.svg", f)
