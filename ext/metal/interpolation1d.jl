
function interpolate1d!(
    ftarget::MtlArray{FT,N},
    xsource::MtlArray{FT,N},
    xtarget::MtlArray{FT,N},
    fsource::MtlArray{FT,N},
    order,
    extrapolate = ClimaInterpolations.Interpolation1D.Flat(),
) where {FT<:Float32,N}
    @assert N ≥ 1
    (nsource, coldims_source...) = size(xsource)
    (ntarget, coldims_target...) = size(xtarget)
    @assert coldims_source == coldims_target
    @assert size(ftarget) == size(xtarget)
    @assert size(fsource) == size(xsource)
    coldims = coldims_source
    colcidxs = CartesianIndices(coldims)
    nitems = length(colcidxs)
    kernel = @metal launch = false interpolate1d_kernel!(
        ftarget,
        xsource,
        xtarget,
        fsource,
        order,
        extrapolate,
        colcidxs,
    )
    nthreads = min(kernel.pipeline.maxTotalThreadsPerThreadgroup, nitems)
    ngroups = convert(Int, cld(nitems, nthreads))

    @metal threads = nthreads groups = ngroups interpolate1d_kernel!(
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
) where {FT<:Float32,N}
    nitems = length(colcidxs)
    tid = thread_position_in_grid_1d()

    if tid ≤ nitems
        colidx = Tuple(colcidxs[tid])
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
