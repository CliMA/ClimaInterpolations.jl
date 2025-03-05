module SplitTrilinearInterpolation
using DocStringExtensions

import ..Interpolation1D: Linear, Flat, interpolate1d!
import ..BilinearInterpolation: Bilinear, interpolatebilinear!, get_dims

struct SplitTrilinear{VS, VT, B, S, W}
    vsource::VS
    vtarget::VT
    bilinear::B
    ordering::S
    workarray::W
end

function SplitTrilinear(
    (vsource, h1source, h2source),
    (vtarget, h1target, h2target);
    vreverse = false,
    ordering = :verticalfirst,
)
    @assert ordering ∈ (:verticalfirst, :horizontalfirst)

    bilinear = Bilinear(h1source, h2source, h1target, h2target)
    nh1source, nh2source, nh1target, nh2target = get_dims(bilinear)
    nvsource = size(vsource, 1)
    nvtarget = size(vtarget, 1)
    workarray =
        ordering == :verticalfirst ?
        similar(vsource, nvtarget, nh1source, nh2source) :
        similar(vsource, nvsource, nh1target, nh2target)

    return SplitTrilinear(vsource, vtarget, bilinear, ordering, workarray)
end

function split_trilinear_interpolation!(
    ftarget,
    splittrilinear::SplitTrilinear,
    fsource;
    vextrapolate = Flat(),
    vreverse = false,
)
    (; vsource, vtarget, bilinear, ordering, workarray) = splittrilinear
    if ordering == :verticalfirst # use vertical first
        interpolate1d!(
            workarray,
            vsource,
            vtarget,
            fsource,
            Linear(),
            vextrapolate,
            reverse = vreverse,
        )
        interpolatebilinear!(ftarget, bilinear, workarray)
    else # use horizontal first
        interpolatebilinear!(workarray, bilinear, fsource)

        interpolate1d!(
            ftarget,
            vsource,
            vtarget,
            workarray,
            Linear(),
            vextrapolate,
            reverse = vreverse,
        )
    end
    return nothing
end

split_trilinear_interpolation!(
    ftarget,
    (vsource, h1source, h2source),
    (vtarget, h1target, h2target),
    fsource;
    vextrapolate = Flat(),
    vreverse = false,
    ordering = :verticalfirst,
) = split_trilinear_interpolation!(
    ftarget,
    SplitTrilinear(
        (vsource, h1source, h2source),
        (vtarget, h1target, h2target),
        vreverse = vreverse,
        ordering = ordering,
    ),
    fsource,
    vextrapolate = vextrapolate,
    vreverse = vreverse,
)

end
