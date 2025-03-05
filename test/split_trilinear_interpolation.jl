using Test
using ClimaInterpolations
import ClimaInterpolations.SplitTrilinearInterpolation:
    split_trilinear_interpolation!
include("utils.jl")

testfunc(x, y, z) = sin(π * x) * cos(π * y) * z

function test_split_trilinear(
    ::Type{DA},
    ::Type{FT},
    (h1min, h1max),
    (h2min, h2max),
    (vmin, vmax),
    (nh1source, nh2source),
    nvsource,
    (nh1target, nh2target),
    nvtarget;
    toler,
) where {DA, FT}
    # build source and target mesh
    h1source, h1target = get_uniform_column_grids(
        DA,
        FT,
        h1min,
        h1max,
        h1min,
        h1max,
        nh1source,
        nh1target,
    )
    h2source, h2target = get_uniform_column_grids(
        DA,
        FT,
        h2min,
        h2max,
        h2min,
        h2max,
        nh2source,
        nh2target,
    )
    vsource, vtarget = get_uniform_column_grids(
        DA,
        FT,
        vmin,
        vmax,
        vmin,
        vmax,
        nvsource,
        nvtarget,
    )
    convert2array = !(DA <: Array)
    vsourcecpu = convert2array ? Array(vsource) : vsource
    h1sourcecpu = convert2array ? Array(fsource) : h1source
    h2sourcecpu = convert2array ? Array(ftarget) : h2source

    fsource =
        testfunc.(
            DA([
                h1sourcecpu[j] for i in 1:nvsource, j in 1:nh1source,
                k in 1:nh2source
            ]),
            DA([
                h2sourcecpu[k] for i in 1:nvsource, j in 1:nh1source,
                k in 1:nh2source
            ]),
            DA([
                vsourcecpu[i] for i in 1:nvsource, j in 1:nh1source,
                k in 1:nh2source
            ]),
        )

    ftarget_vfirst = DA{FT}(undef, nvtarget, nh1target, nh2target)

    vtargetcpu = convert2array ? Array(vtarget) : vtarget
    h1targetcpu = convert2array ? Array(h1target) : h1target
    h2targetcpu = convert2array ? Array(h2target) : h2target

    ftargetexact =
        testfunc.(
            DA([
                h1targetcpu[j] for i in 1:nvtarget, j in 1:nh1target,
                k in 1:nh2target
            ]),
            DA([
                h2targetcpu[k] for i in 1:nvtarget, j in 1:nh1target,
                k in 1:nh2target
            ]),
            DA([
                vtargetcpu[i] for i in 1:nvtarget, j in 1:nh1target,
                k in 1:nh2target
            ]),
        )

    split_trilinear_interpolation!(
        ftarget_vfirst,
        (vsource, h1source, h2source),
        (vtarget, h1target, h2target),
        fsource,
    )
    @show "using verticalfirst option"
    @show size(ftarget_vfirst)
    @show size(ftargetexact)
    @show maximum(abs.(ftarget_vfirst .- ftargetexact))

    ftarget_hfirst = DA{FT}(undef, nvtarget, nh1target, nh2target)

    split_trilinear_interpolation!(
        ftarget_hfirst,
        (vsource, h1source, h2source),
        (vtarget, h1target, h2target),
        fsource,
        ordering = :horizontalfirst,
    )

    @show "using horizontalfirst option"
    @show size(ftarget_hfirst)
    @show size(ftargetexact)
    @show maximum(abs.(ftarget_hfirst .- ftargetexact))
    return nothing
end

get_dims_split_trilinear(::Type{FT}) where {FT} = (
    (FT(0), FT(3π)),
    (FT(0), FT(2π)),
    (FT(0), FT(1)),
    (1280, 640),
    (1200, 600),
    (128, 256),
    FT(0.0002),
)

FT = Float32
DA = Array
(h1min, h1max),
(h2min, h2max),
(vmin, vmax),
(nh1source, nh2source),
(nh1target, nh2target),
(nvsource, nvtarget),
toler = get_dims_split_trilinear(FT)

test_split_trilinear(
    DA,
    FT,
    (h1min, h1max),
    (h2min, h2max),
    (vmin, vmax),
    (nh1source, nh2source),
    nvsource,
    (nh1target, nh2target),
    nvtarget;
    toler,
)
