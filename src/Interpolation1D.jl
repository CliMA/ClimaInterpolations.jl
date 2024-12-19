module Interpolation1D

include("stencils1d.jl")

"""
    interpolate(xpts, fval, xtarg)

Interpolate `fval` defined on `xpts` at `xtarg`.
Reference:
1). Berrut, J., and Trefethen, L. N., 2004. Barycentric Lagrange Interpolation, SIAM REVIEW
Vol. 46, No. 3, pp. 501–517.
DOI. 10.1137/S0036144502417715
"""
Base.@pure function interpolate(
    xpts::AbstractArray{FT,1},
    fval::AbstractArray{FT,1},
    xtarg::FT,
) where {FT<:AbstractFloat}
    length(xpts) == 1 && return fval[1] # flat extrapolation
    l, tot, mi, tol = FT(1), -FT(0), 0, (10 * eps(FT))
    @inbounds begin
        for i in eachindex(xpts)
            dx = xtarg - xpts[i]
            if abs(dx) < tol
                mi = i
                break
            else
                l *= dx
            end
        end
        if mi == 0
            for i in eachindex(xpts)
                xi = xpts[i]
                wi = FT(1)
                for j in eachindex(xpts)
                    if j ≠ i
                        wi *= (xi - xpts[j])
                    end
                end
                tot += fval[i] * l / ((xtarg - xi) * wi)
            end
        else
            tot = fval[mi]
        end
    end
    return tot
end

"""
    interpolate1d!(
        ftarget::AbstractArray{FT,N},
        xsource::AbstractArray{FT,N},
        xtarget::AbstractArray{FT,N},
        fsource::AbstractArray{FT,N},
        order,
        extrapolate = Flat(),
    ) where {FT,N}

Interpolate `fsource`, defined on grid `xsource`, onto the `xtarget` grid.
Here the source grid `xsource` is an N-dimensional array of columns.
The first dimension is assumed to be the column dimension. 
Each column can have a different grid.
"""
function interpolate1d!(
    ftarget::AbstractArray{FT,N},
    xsource::AbstractArray{FT,N},
    xtarget::AbstractArray{FT,N},
    fsource::AbstractArray{FT,N},
    order,
    extrapolate = Flat(),
) where {FT,N}
    @assert N ≥ 1
    (nsource, coldims_source...) = size(xsource)
    (ntarget, coldims_target...) = size(xtarget)
    @assert coldims_source == coldims_target
    @assert size(ftarget) == size(xtarget)
    @assert size(fsource) == size(xsource)
    coldims = coldims_source
    colcidxs = CartesianIndices(coldims)
    @inbounds begin
        for colcidx in colcidxs
            colidx = Tuple(colcidx)
            first = 1
            for i1 = 1:ntarget
                (st, en) = get_stencil(
                    order,
                    view(xsource, :, colidx...),
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
    end
    return nothing
end

end
