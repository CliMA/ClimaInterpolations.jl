module BilinearInterpolation
using DocStringExtensions
import ..Interpolation1D: get_stencil, Linear, Flat

"""
    Bilinear{V,I}

This struct stores the source and target grids for bilinear interpolation on a rectangular grid.
Indexes specifying the location of the target points in the source grid are also stored for 
enabling efficient implementation of bilinear interpolation.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct Bilinear{V,I}
    "Source grid in `x` direction `(nsourcex)`"
    sourcex::V
    "Source grid in `y` direction `(nsourcey)`"
    sourcey::V
    "Target grid in `x` direction `(ntargetx)`"
    targetx::V
    "Target grid in `y` direction `(ntargety)`"
    targety::V
    "left index of location of a target `x` grid point in source `x` grid"
    startx::I
    "left index of location of a target `y` grid point in source `y` grid"
    starty::I
end

get_dims(b::Bilinear) =
    (length(b.sourcex), length(b.sourcey), length(b.targetx), length(b.targety))

function Bilinear(sourcex, sourcey, targetx, targety)
    nsourcex, nsourcey = length(sourcex), length(sourcey)
    ntargetx, ntargety = length(targetx), length(targety)
    startx = similar(targetx, Int)
    starty = similar(targety, Int)
    set_source_range!(startx, sourcex, targetx)
    set_source_range!(starty, sourcey, targety)
    return Bilinear(sourcex, sourcey, targetx, targety, startx, starty)
end

function set_source_range!(
    start::AbstractVector{I},
    source::AbstractVector{FT},
    target::AbstractVector{FT},
) where {I,FT}
    order = Linear()
    first = 1
    @inbounds begin
        for (ixt, x) in enumerate(target)
            st = get_stencil_bilinear1d(order, source, x, first = first)
            first = st
            start[ixt] = st
        end
    end
    return nothing
end

"""
    interpolatebilinear!(
        ftarget::AbstractArray{FT,N},
        bilinear::B,
        fsource::AbstractArray{FT,N},
    ) where {FT,N,B}

Interpolate `fsource`, defined on source grid, onto the target grid.
The horizontal source and targets are define in `bilinear`.
Here `fsource` is an N-dimensional array where the last two dimensions are assumed to be horizontal dimensions.

For example, `fsource` can be of size `[n1, n2..., nx, ny]`, where `nx` and `ny` are the horizontal dimensions.
Single horizontal level is also supported.
The number of horizontal levels should be same for both source and target arrays.
"""
function interpolatebilinear!(
    ftarget::AbstractArray{FT,N},
    bilinear::B,
    fsource::AbstractArray{FT,N},
) where {FT,N,B}
    @assert N ≥ 2
    (leveldimssource..., nxs, nys) = size(fsource)
    (leveldimstarget..., nxt, nyt) = size(ftarget)

    (; sourcex, sourcey, targetx, targety) = bilinear
    nsourcex, nsourcey, ntargetx, ntargety = get_dims(bilinear)

    @assert leveldimssource == leveldimstarget
    @assert nxs == nsourcex && nys == nsourcey
    @assert nxt == ntargetx && nyt == ntargety

    leveldims = leveldimssource
    levelcidxs = CartesianIndices(leveldims)

    @inbounds begin
        for (iyt, y) in enumerate(targety)
            sty = bilinear.starty[iyt]
            ly = sourcey[sty+1] - sourcey[sty]
            dy2 = sourcey[sty+1] - y
            dy1 = y - sourcey[sty]

            for (ixt, x) in enumerate(targetx)
                stx = bilinear.startx[ixt]
                lx = sourcex[stx+1] - sourcex[stx]
                dx2 = sourcex[stx+1] - x
                dx1 = x - sourcex[stx]
                fac = FT(1) / (lx * ly)

                for levelcidx in levelcidxs
                    levelidx = Tuple(levelcidx)
                    ftarget[levelidx..., ixt, iyt] =
                        (
                            dx2 * dy2 * fsource[levelidx..., stx, sty] +
                            dx1 * dy2 * fsource[levelidx..., stx+1, sty] +
                            dx2 * dy1 * fsource[levelidx..., stx, sty+1] +
                            dx1 * dy1 * fsource[levelidx..., stx+1, sty+1]
                        ) * fac
                end
            end
        end
    end
    return nothing
end

function get_stencil_bilinear1d(alg::Linear, source, target; first = 1)
    st, _ = get_stencil(alg, source, target, first = first)
    return min(st, length(source) - 1)
end

end