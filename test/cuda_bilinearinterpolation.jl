import ClimaInterpolations.BilinearInterpolation: Bilinear, interpolatebilinear!
using Test
using BenchmarkTools
using Statistics
using CUDA
include("utils.jl")

testfunc(x, y) = sin(π * x) * cos(π * y)
testfunc(x, y, z) = sin(π * x) * cos(π * y) * z

function test_single_level(
    ::Type{DA},
    ::Type{FT},
    (xmin, xmax),
    (ymin, ymax),
    (nsourcex, nsourcey),
    (ntargetx, ntargety);
    toler,
) where {DA,FT}
    trial_constructor = nothing
    trial_interpolator = nothing
    @testset "bilinear interpolation on single level with $FT, nsource = ($nsourcex, $nsourcey), ntarget = ($ntargetx, $ntargety)" begin
        # build source and target mesh
        xsource, xtarget =
            get_uniform_column_grids(DA, FT, xmin, xmax, xmin, xmax, nsourcex, ntargetx)
        ysource, ytarget =
            get_uniform_column_grids(DA, FT, ymin, ymax, ymin, ymax, nsourcey, ntargety)
        bilinear = Bilinear(xsource, ysource, xtarget, ytarget)
        trial_constructor = @benchmark Bilinear($xsource, $ysource, $xtarget, $ytarget)
        # build fsource
        xsourcecpu, ysourcecpu = Array(xsource), Array(ysource)
        sourcemesh = (
            x = DA([xsourcecpu[i] for i = 1:nsourcex, j = 1:nsourcey]),
            y = DA([ysourcecpu[j] for i = 1:nsourcex, j = 1:nsourcey]),
        )
        xtargetcpu, ytargetcpu = Array(xtarget), Array(ytarget)
        targetmesh = (
            x = DA([xtargetcpu[i] for i = 1:ntargetx, j = 1:ntargety]),
            y = DA([ytargetcpu[j] for i = 1:ntargetx, j = 1:ntargety]),
        )
        fsource = testfunc.(sourcemesh.x, sourcemesh.y)
        ftargetexact = testfunc.(targetmesh.x, targetmesh.y)
        # allocate ftarget
        ftarget = DA{FT}(undef, ntargetx, ntargety)
        # use bilinear interpolation
        interpolatebilinear!(ftarget, bilinear, fsource)
        l∞error = maximum(abs.((ftarget.-ftargetexact)[:]))
        @show l∞error
        @test l∞error ≤ toler
        if DA <: CuArray
            trial_interpolator =
                @benchmark CUDA.@sync interpolatebilinear!($ftarget, $bilinear, $fsource)
        else
            trial_interpolator =
                @benchmark interpolatebilinear!($ftarget, $bilinear, $fsource)
        end
    end
    return (trial_constructor, trial_interpolator)
end

function test_multilevel(
    ::Type{DA},
    ::Type{FT},
    (xmin, xmax),
    (ymin, ymax),
    (zmin, zmax),
    (nsourcex, nsourcey),
    (ntargetx, ntargety),
    nlevels;
    toler,
) where {DA,FT}
    trial_constructor = nothing
    trial_interpolator = nothing
    @testset "bilinear interpolation on $nlevels levels with $FT, nsource = ($nsourcex, $nsourcey), ntarget = ($ntargetx, $ntargety)" begin
        # build source and target mesh
        xsource, xtarget =
            get_uniform_column_grids(DA, FT, xmin, xmax, xmin, xmax, nsourcex, ntargetx)
        ysource, ytarget =
            get_uniform_column_grids(DA, FT, ymin, ymax, ymin, ymax, nsourcey, ntargety)
        z, _ = get_uniform_column_grids(DA, FT, zmin, zmax, zmin, zmax, nlevels, nlevels)
        bilinear = Bilinear(xsource, ysource, xtarget, ytarget)
        trial_constructor = @benchmark Bilinear($xsource, $ysource, $xtarget, $ytarget)
        # build fsource
        xsourcecpu, ysourcecpu, zcpu = Array(xsource), Array(ysource), Array(z)
        sourcemesh = (
            x = DA([xsourcecpu[j] for i = 1:nlevels, j = 1:nsourcex, k = 1:nsourcey]),
            y = DA([ysourcecpu[k] for i = 1:nlevels, j = 1:nsourcex, k = 1:nsourcey]),
            z = DA([zcpu[i] for i = 1:nlevels, j = 1:nsourcex, k = 1:nsourcey]),
        )
        xtargetcpu, ytargetcpu = Array(xtarget), Array(ytarget)
        targetmesh = (
            x = DA([xtargetcpu[j] for i = 1:nlevels, j = 1:ntargetx, k = 1:ntargety]),
            y = DA([ytargetcpu[k] for i = 1:nlevels, j = 1:ntargetx, k = 1:ntargety]),
            z = DA([zcpu[i] for i = 1:nlevels, j = 1:ntargetx, k = 1:ntargety]),
        )
        fsource = testfunc.(sourcemesh.x, sourcemesh.y, sourcemesh.z)
        ftargetexact = testfunc.(targetmesh.x, targetmesh.y, targetmesh.z)
        # allocate ftarget
        ftarget = DA{FT}(undef, nlevels, ntargetx, ntargety)
        # use bilinear interpolation
        interpolatebilinear!(ftarget, bilinear, fsource)
        l∞error = maximum(abs.((ftarget.-ftargetexact)[:]))
        @show l∞error
        @test l∞error ≤ toler
        if DA <: CuArray
            trial_interpolator =
                @benchmark CUDA.@sync interpolatebilinear!($ftarget, $bilinear, $fsource)
        else
            trial_interpolator =
                @benchmark interpolatebilinear!($ftarget, $bilinear, $fsource)
        end
    end
    return (trial_constructor, trial_interpolator)
end
#=
get_dims_singlelevel(::Type{FT}) where {FT} =
    ((FT(0), FT(3π)), (FT(0), FT(2π)), (200, 150), (300, 400), FT(0.005))
get_dims_multilevel(::Type{FT}) where {FT} = (
    (FT(0), FT(3π)),
    (FT(0), FT(2π)),
    (FT(0), FT(1)),
    (200, 150),
    (300, 400),
    2048,
    FT(0.005),
)
=#
get_dims_singlelevel(::Type{FT}) where {FT} =
    ((FT(0), FT(3π)), (FT(0), FT(2π)), (2560, 1280), (2400, 1200), FT(0.00005))
get_dims_multilevel(::Type{FT}) where {FT} = (
    (FT(0), FT(3π)),
    (FT(0), FT(2π)),
    (FT(0), FT(1)),
    (2560, 1280),
    (2400, 1200),
    128,
    FT(0.00005),
)
@show "--------------CPU benchmarks----------------"
xrange, yrange, nsource, ntarget, toler = get_dims_singlelevel(Float32)
(trial_sl_cpu_ft32_cons, trial_sl_cpu_ft32_interp) =
    test_single_level(Array, Float32, xrange, yrange, nsource, ntarget, toler = toler)
println(
    "interpolation time = $(Statistics.median(trial_sl_cpu_ft32_interp)) on CPU for single level with Float32\nConstructor time = $(Statistics.median(trial_sl_cpu_ft32_cons))",
)
@show "--------------------------------------------"

xrange, yrange, nsource, ntarget, toler = get_dims_singlelevel(Float64)
trial_sl_cpu_ft64_cons, trial_sl_cpu_ft64_interp =
    test_single_level(Array, Float64, xrange, yrange, nsource, ntarget, toler = toler)
println(
    "interpolation time = $(Statistics.median(trial_sl_cpu_ft64_interp)) on CPU for single level with Float64\nConstructor time = $(Statistics.median(trial_sl_cpu_ft64_cons))",
)
@show "--------------------------------------------"
xrange, yrange, zrange, nsource, ntarget, nlevels, toler = get_dims_multilevel(Float32)
trial_ml_cpu_ft32_cons, trial_ml_cpu_ft32_interp = test_multilevel(
    Array,
    Float32,
    xrange,
    yrange,
    zrange,
    nsource,
    ntarget,
    nlevels,
    toler = toler,
)
println(
    "interpolation time = $(Statistics.median(trial_ml_cpu_ft32_interp)) on CPU for $nlevels levels with Float32\nConstructor time = $(Statistics.median(trial_ml_cpu_ft32_cons))",
)
@show "--------------------------------------------"
xrange, yrange, zrange, nsource, ntarget, nlevels, toler = get_dims_multilevel(Float64)
trial_ml_cpu_ft64_cons, trial_ml_cpu_ft64_interp = test_multilevel(
    Array,
    Float64,
    xrange,
    yrange,
    zrange,
    nsource,
    ntarget,
    nlevels,
    toler = toler,
)
println(
    "interpolation time = $(Statistics.median(trial_ml_cpu_ft64_interp)) on CPU for $nlevels levels with Float64\nConstructor time = $(Statistics.median(trial_ml_cpu_ft64_cons))",
)
@show "--------------CUDA benchmarks---------------"
xrange, yrange, nsource, ntarget, toler = get_dims_singlelevel(Float32)
trial_sl_cuda_ft32_cons, trial_sl_cuda_ft32_interp =
    test_single_level(CuArray, Float32, xrange, yrange, nsource, ntarget, toler = toler)
println(
    "interpolation time = $(Statistics.median(trial_sl_cuda_ft32_interp)) on NVIDIA GPU for single level with Float32\nConstructor time = $(Statistics.median(trial_sl_cuda_ft32_cons))",
)
@show "--------------------------------------------"
xrange, yrange, zrange, nsource, ntarget, nlevels, toler = get_dims_multilevel(Float32)
trial_ml_cuda_ft32_cons, trial_ml_cuda_ft32_interp = test_multilevel(
    CuArray,
    Float32,
    xrange,
    yrange,
    zrange,
    nsource,
    ntarget,
    nlevels,
    toler = toler,
)
println(
    "interpolation time = $(Statistics.median(trial_ml_cuda_ft32_interp)) on NVIDIA GPU for $nlevels levels with Float32\nConstructor time = $(Statistics.median(trial_ml_cuda_ft32_cons))",
)
@show "--------------------------------------------"
