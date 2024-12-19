
function interpolate1d!(
    ftarget::CuArray{FT,N},
    xsource::CuArray{FT,N},
    xtarget::CuArray{FT,N},
    fsource::CuArray{FT,N},
    order,
    extrapolate = ClimaInterpolations.Interpolation1D.Flat(),
) where {FT,N}
    @assert N ≥ 1
    (nsource, coldims_source...) = size(xsource)
    (ntarget, coldims_target...) = size(xtarget)
    @assert coldims_source == coldims_target
    @assert size(ftarget) == size(xtarget)
    @assert size(fsource) == size(xsource)
    coldims = coldims_source
    colcidxs = CartesianIndices(coldims)
    nitems = length(colcidxs)
    kernel = @cuda launch = false interpolate1d_kernel!(
        ftarget,
        xsource,
        xtarget,
        fsource,
        order,
        extrapolate,
        colcidxs,
    )
    kernel_config = CUDA.launch_configuration(kernel.fun)
    nthreads = min(kernel_config.threads, nitems)
    nblocks = convert(Int, cld(nitems, nthreads))

    @cuda threads = nthreads blocks = nblocks interpolate1d_kernel!(
        ftarget,
        xsource,
        xtarget,
        fsource,
        order,
        extrapolate,
        colcidxs,
    )
    return nothing
end

function interpolate1d_kernel!(
    ftarget::AbstractArray{FT,N},
    xsource::AbstractArray{FT,N},
    xtarget::AbstractArray{FT,N},
    fsource::AbstractArray{FT,N},
    order,
    extrapolate,
    colcidxs,
) where {FT,N}
    nitems = length(colcidxs)
    gid = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if gid ≤ nitems
        colidx = Tuple(colcidxs[gid])
        nsource = size(xsource, 1)
        ntarget = size(xtarget, 1)
        first = 1

        for i1 = 1:ntarget
            (st, en) = get_stencil(
                order,
                view(xsource, 1:nsource, colidx...),
                xtarget[i1, colidx...],
                first = first,
                extrapolate = extrapolate,
            )
            first = st
            ftarget[i1, colidx...] = interpolate(
                view(xsource, st:en, colidx...),
                view(fsource, st:en, colidx...),
                xtarget[i1, colidx...],
            )
        end
    end
    return nothing
end
