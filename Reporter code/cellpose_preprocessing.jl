using AbstractFFTs
using Compat
using FFTW
using Images
using TiffImages
using TensorOperations
using StatsBase
using CoordinateTransformations

function phase_offset(source, target; kwargs...)
    plan = plan_fft(source)
    return phase_offset(plan, plan * source, plan * target; kwargs...)
end

function phase_offset(plan, source_freq::AbstractArray{<:Complex{T}},
                        target_freq; upsample_factor = 1, 
                        normalize = false) where {T}
    
    image_product = @. source_freq * conj(target_freq)
    if normalize
        @. image_product /= max(abs(image_product), eps(T))
    end
    if isone(upsample_factor)
        cross_correlation = ifft!(image_product)
    else
        cross_correlation = plan \ image_product
    end
    maxima, maxidx = @compat findmax(abs, cross_correlation)
    shape = size(source_freq)
    midpoints = map(ax -> (first(ax) + last(ax)) / T(2), axes(source_freq))
    idxoffset = map(first, axes(cross_correlation))
    shift = @. T(ifelse(maxidx.I > midpoints, maxidx.I - shape, maxidx.I) - idxoffset)

    isone(upsample_factor) &&
        return (; shift, calculate_stats(maxima, source_freq, target_freq)...)

    shift = @. round(shift * upsample_factor) / T(upsample_factor)
    upsample_region_size = ceil(upsample_factor * T(1.5))
    dftshift = div(upsample_region_size, 2)
    sample_region_offset = @. dftshift - shift * upsample_factor
    cross_correlation = upsampled_dft(
        image_product,
        upsample_region_size,
        upsample_factor,
        sample_region_offset,
    )
    maxima, maxidx = @compat findmax(abs, cross_correlation)
    shift = @. shift + (maxidx.I - dftshift - idxoffset) / T(upsample_factor)

    stats = calculate_stats(maxima, source_freq, target_freq)
    return (; shift, stats...)
end

function contract_tensors(kernel, data)
    data = conj.(data)
    if ndims(data) == 2
        @tensor begin
            _data[i,j] := kernel[i,:] * data[j,:]
        end
    else
        @tensor begin
            _data[i,j,k] := kernel[i,:] * data[j,k,:]
        end
    end
    return _data 
end

function upsampled_dft(data::AbstractArray{T}, region_size, upsample_factor,
                            offsets) where {T<:Complex}
    
    shiftrange = 1:region_size
    sample_rate = inv(T(upsample_factor))
    idxoffsets = map(first, axes(data))
    shape = size(data)
    offsets = offsets
    _data = copy(data)

    for (k,(dim, offset, idxoffset)) in enumerate(zip(reverse(shape), 
                                                      reverse(offsets), 
                                                      reverse(idxoffsets)))
        if iseven(k)
            sign = 1
        else
            sign = -1
        end
        freqs = fftfreq(dim, sample_rate)
        kernel = @. cis(sign*T(2π) * (shiftrange - offset - idxoffset) * freqs')
        _data = contract_tensors(kernel, _data) 
    end
    return _data
end

function calculate_stats(crosscor_maxima, source_freq, target_freq)
    source_amp = mean(abs2, source_freq)
    target_amp = mean(abs2, target_freq)
    error = 1 - abs2(crosscor_maxima) / (source_amp * target_amp)
    phasediff = atan(imag(crosscor_maxima), real(crosscor_maxima))
    return (; error, phasediff)
end

function crop(img_stack)
    # get sizes
    nx, ny, nz, nt = size(img_stack)
    # tiny 1-D masks for each axis
    mask_i = falses(nx)
    mask_j = falses(ny)
    mask_z = falses(nz)
    # background value
    oneval = one(eltype(img_stack))

    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        # check if this (i,j,k) is never background across all t
        is_forever = true
        for t in 1:nt
            if img_stack[i, j, k, t] == oneval
                is_forever = false
                break
            end
        end
        if is_forever
            mask_i[i] = true
            mask_j[j] = true
            mask_z[k] = true
        end
    end

    # find first/last true in each axis-mask
    i1, i2 = findfirst(mask_i), findlast(mask_i)
    j1, j2 = findfirst(mask_j), findlast(mask_j)
    z1, z2 = findfirst(mask_z), findlast(mask_z)

    # return a SubArray view (no copy)
    return @view img_stack[i1:i2, j1:j2, z1:z2, :]
end

function register!(img_stack, other, registered_stack, registered_other, nframes)       
    # Register all images to the first image in the stack
    shifts = (0.0,0.0,0.0)
    for t in 2:nframes
        @views reference = img_stack[:,:,:,t-1]
        @views moving = img_stack[:,:,:,t]
        @views moving_other = other[:,:,:,t]
        shift, _, _ = phase_offset(reinterpret(Float32, reference), reinterpret(Float32, moving), upsample_factor=10)
        shifts = (shifts[1] + shift[1], shifts[2] + shift[2], shifts[3] + shift[3]) 
        shift = Translation(-1*shifts[1], -1*shifts[2], 0)
        registered_stack[:,:,:,t] = warp(moving, shift, axes(moving), 1)
        registered_other[:,:,:,t] = warp(moving_other, shift, axes(moving_other), 1)
    end
    return nothing
end

function sep_z_t(timeseries)
    height, width, zt = size(timeseries)
    ImageJ_metadata = first(ifds(timeseries))[TiffImages.IMAGEDESCRIPTION].data
    slice_loc = findfirst("slices=", ImageJ_metadata)
    frames_loc = findfirst("frames=", ImageJ_metadata)
    hyperstack_loc = findfirst("hyperstack=", ImageJ_metadata)
    slices = tryparse(Int, ImageJ_metadata[(slice_loc[7]+1):(frames_loc[1]-1)])
    frames = tryparse(Int, ImageJ_metadata[(frames_loc[7]+1):(hyperstack_loc[1]-1)])
    if slices == nothing
        error("Could not parse the number of slices from the metadata")
    end
    if frames == nothing
        error("Could not parse the number of frames from the metadata")
    end
    return reshape(timeseries, (height, width, slices, frames))
end

function main()
    # Load images
    if !isdir("Cells")
        mkdir("Cells")
    end
    if !isdir("Reporter")
        mkdir("Reporter")
    end
    cells_path = "cells_noback.tif"
    other_path = "rbmA_noback.tif"
    cells = TiffImages.load(cells_path)
	cells = sep_z_t(cells) 
    other = TiffImages.load(other_path)
	other = sep_z_t(other) 
    registered_cells = copy(cells)
    registered_other = copy(other)
    height, width, depth, nframes = size(cells)
    register!(cells, other, registered_cells, registered_other, nframes)
	cells = nothing
	other = nothing
    cropped_cells = crop(registered_cells)
    cropped_other = crop(registered_other)
    for t in 1:nframes
        @views TiffImages.save("Cells/"*"processed_cells_$(t).tif", cropped_cells[:,:,:,t])
        @views TiffImages.save("Reporter/"*"processed_other_$(t).tif", cropped_other[:,:,:,t])
    end
end

main()
