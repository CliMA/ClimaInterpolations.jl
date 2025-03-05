module SplitTrilinearInterpolation
using DocStringExtensions

import ..Interpolation1D: Linear, Flat, interpolate1d!
import ..BilinearInterpolation: Bilinear, interpolatebilinear!, get_dims

struct SplitTrilinearHV{VS, VT, HB, W1, W2}
    vsource::VS
    vtarget::VT
    bilinear::HB
    fsourceintermediate::W1
    vsourceintermediate::W2
end

function SplitTrilinearHV(
    (vsource, h1source, h2source),
    (vtarget, h1target, h2target);
)
    bilinear = Bilinear(h1source, h2source, h1target, h2target)
    nvsource = size(vsource, 1)
    _, _, nh1target, nh2target = get_dims(bilinear)
    fsourceintermediate = similar(vsource, nvsource, nh1target, nh2target)
    vsourceintermediate =
        ndims(vsource) == 1 ? vsource :
        similar(vsource, nvsource, nh1target, nh2target)
    ndims(vsource) == 1 ||
        interpolatebilinear!(vsourceintermediate, bilinear, vsource)
    return SplitTrilinearHV(
        vsource,
        vtarget,
        bilinear,
        fsourceintermediate,
        vsourceintermediate,
    )
end

function split_trilinear_interpolation!(
    ftarget,
    splittrilinear::SplitTrilinearHV,
    fsource;
    vextrapolate = Flat(),
    vreverse = false,
)
    (; vtarget, bilinear, fsourceintermediate, vsourceintermediate) =
        splittrilinear
    # applies bilinear interpolation at each horizontal level
    interpolatebilinear!(fsourceintermediate, bilinear, fsource)
    # then apply linear interpolation in the vertical direction
    interpolate1d!(
        ftarget,
        vsourceintermediate,
        vtarget,
        fsourceintermediate,
        Linear(),
        vextrapolate,
        reverse = vreverse,
    )
    return nothing
end

struct SplitTrilinearVH{VS, VT, HB, W1}
    vsource::VS
    vtarget::VT
    bilinear::HB
    fsourceintermediate::W1
end

function SplitTrilinearVH(
    (vsource, h1source, h2source),
    (vtarget, h1target, h2target);
)
    bilinear = Bilinear(h1source, h2source, h1target, h2target)
    nvtarget = size(vtarget, 1)
    nh1source, nh2source, _, _ = get_dims(bilinear)
    fsourceintermediate = similar(vsource, nvtarget, nh1source, nh2source)
    return SplitTrilinearVH(vsource, vtarget, bilinear, fsourceintermediate)
end

function split_trilinear_interpolation!(
    ftarget,
    splittrilinear::SplitTrilinearVH,
    fsource;
    vextrapolate = Flat(),
    vreverse = false,
)
    (; vsource, vtarget, bilinear, fsourceintermediate) = splittrilinear
    interpolate1d!(
        fsourceintermediate,
        vsource,
        vtarget,
        fsource,
        Linear(),
        vextrapolate,
        reverse = vreverse,
    )
    interpolatebilinear!(ftarget, bilinear, fsourceintermediate)
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
) =
    ordering == :verticalfirst ?
    split_trilinear_interpolation!(
        ftarget,
        SplitTrilinearVH(
            (vsource, h1source, h2source),
            (vtarget, h1target, h2target),
        ),
        fsource,
        vextrapolate = vextrapolate,
        vreverse = vreverse,
    ) :
    split_trilinear_interpolation!(
        ftarget,
        SplitTrilinearHV(
            (vsource, h1source, h2source),
            (vtarget, h1target, h2target),
        ),
        fsource,
        vextrapolate = vextrapolate,
        vreverse = vreverse,
    )

end
