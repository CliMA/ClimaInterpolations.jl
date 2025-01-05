using CUDA
import ClimaInterpolations.Interpolation1D:
    interpolate, Linear, get_stencil, interpolate1d!, Flat, LinearExtrapolation
using Test
using BenchmarkTools
using Statistics

include("utils.jl")

function test_single_column(
    ::Type{DA},
    ::Type{FT},
    xmin,
    xmax,
    nsource,
    ntarget;
    xmintarg = xmin,
    xmaxtarg = xmax,
    extrapolation = Flat(),
) where {DA,FT}
    trial_data = nothing
    @testset "1D linear interpolation on single column with $FT" begin
        toler = FT(0.003)
        order = Linear()
        xsource, xtarget = get_uniform_column_grids(
            DA,
            FT,
            xmin,
            xmax,
            xmintarg,
            xmaxtarg,
            nsource,
            ntarget,
        )
        fsource = sin.(xsource) # function defined on source grid
        ftarget = DA(zeros(FT, ntarget)) # allocated function on target grid
        interpolate1d!(ftarget, xsource, xtarget, fsource, order, extrapolation)
        diff = maximum(
            abs.(ftarget .- sin.(xtarget)) .* (xtarget .≤ xmax) .* (xtarget .≥ xmin),
        )
        @test diff ≤ toler
        converttoarray = !(DA <: Array)
        xtarget = converttoarray ? DA(xtarget) : xtarget
        ftarget = converttoarray ? DA(ftarget) : ftarget
        fsource = converttoarray ? DA(fsource) : fsource
        # test extrapolation
        if xmintarg < xmin || xmaxtarg > xmax
            if extrapolation == Flat()
                left_boundary_pass = true
                right_boundary_pass = true
                for i = 1:length(xtarget)
                    if xtarget[i] < xmin
                        left_boundary_pass = ftarget[i] == fsource[1]
                    end
                    if xtarget[i] > xmax
                        right_boundary_pass = ftarget[i] == fsource[end]
                    end
                end
                @testset "testing Flat extrapolation" begin
                    @test left_boundary_pass
                    @test right_boundary_pass
                end
            end
        end
        if DA <: CuArray
            trial_data = @benchmark CUDA.@sync interpolate1d!(
                $ftarget,
                $xsource,
                $xtarget,
                $fsource,
                $order,
                $extrapolation,
            )
        else
            trial_data = @benchmark interpolate1d!(
                $ftarget,
                $xsource,
                $xtarget,
                $fsource,
                $order,
                $extrapolation,
            )
        end
    end
    return trial_data
end

function test_multiple_columns(
    ::Type{DA},
    ::Type{FT},
    xmin,
    xmax,
    nsource,
    ntarget,
    nlon,
    nlat;
    xmintarg = xmin,
    xmaxtarg = xmax,
    extrapolation = Flat(),
) where {DA,FT}
    trial_data = nothing
    @testset "1D linear interpolation on multiple columns with $FT on $DA" begin
        toler = FT(0.003)
        xsource, xtarget = get_uniform_column_grids(
            Array,
            FT,
            xmin,
            xmax,
            xmintarg,
            xmaxtarg,
            nsource,
            ntarget,
        )

        xsourcecols = DA(repeat(xsource, 1, nlon, nlat))
        xtargetcols = DA(repeat(xtarget, 1, nlon, nlat))
        fsourcecols = DA(sin.(xsourcecols))
        ftargetcols = DA(zeros(FT, ntarget, nlon, nlat))
        order = Linear()

        interpolate1d!(
            ftargetcols,
            xsourcecols,
            xtargetcols,
            fsourcecols,
            order,
            extrapolation,
        )
        diff = maximum(abs.(ftargetcols .- sin.(xtargetcols))[:])
        @test diff ≤ toler
        if DA <: CuArray
            trial_data = @benchmark CUDA.@sync interpolate1d!(
                $ftargetcols,
                $xsourcecols,
                $xtargetcols,
                $fsourcecols,
                $order,
                $extrapolation,
            )
        else
            trial_data = @benchmark interpolate1d!(
                $ftargetcols,
                $xsourcecols,
                $xtargetcols,
                $fsourcecols,
                $order,
                $extrapolation,
            )
        end
    end
    return trial_data
end


get_dims_singlecol(::Type{FT}) where {FT} = (FT(0), FT(2π), 150, 200)
get_dims_multicol(::Type{FT}) where {FT} = (FT(0), FT(2π), 150, 200, 2560, 1280)

println("Running tests and benchmarks on CPU")
println("********************************************************")

# single column linear interpolation tests without extrapolation
trial_sc_cpu_ft32 = test_single_column(Array, Float32, get_dims_singlecol(Float32)...)
println(
    "median time = $(Statistics.median(trial_sc_cpu_ft32)) on CPU for single column with Float32",
)
trial_sc_cpu_ft64 = test_single_column(Array, Float64, get_dims_singlecol(Float64)...)
println(
    "median time = $(Statistics.median(trial_sc_cpu_ft64)) on CPU for single column with Float64",
)

# single column linear interpolation tests with Flat extrapolation
xmin, xmax, nsource, ntarget = get_dims_singlecol(Float32)
test_single_column(
    Array,
    Float32,
    xmin,
    xmax,
    nsource,
    ntarget,
    xmintarg = xmin - 1.0,
    xmaxtarg = xmax + 1.0,
    extrapolation = Flat(),
)
xmin, xmax, nsource, ntarget = get_dims_singlecol(Float64)
test_single_column(
    Array,
    Float64,
    xmin,
    xmax,
    nsource,
    ntarget,
    xmintarg = xmin - 1.0,
    xmaxtarg = xmax + 1.0,
    extrapolation = Flat(),
)
# multiple column linear interpolation tests without extrapolation
xmin, xmax, nsource, ntarget, nlon, nlat = get_dims_multicol(Float32)
trial_mc_cpu_ft32 =
    test_multiple_columns(Array, Float32, xmin, xmax, nsource, ntarget, nlon, nlat)
println(
    "median time = $(Statistics.median(trial_mc_cpu_ft32)) on CPU for $nlon x $nlat columns with Float32",
)

xmin, xmax, nsource, ntarget, nlon, nlat = get_dims_multicol(Float64)
trial_mc_cpu_ft64 =
    test_multiple_columns(Array, Float64, xmin, xmax, nsource, ntarget, nlon, nlat)
println(
    "median time = $(Statistics.median(trial_mc_cpu_ft32)) on CPU for $nlon x $nlat columns with Float64",
)

println("Running tests and benchmarks on NVIDIA GPU")
println("********************************************************")

# single column linear interpolation tests without extrapolation
xmin, xmax, nsource, ntarget = get_dims_singlecol(Float32)
trial_sc_cuda_gpu_ft32 = test_single_column(CuArray, Float32, xmin, xmax, nsource, ntarget)
println(
    "median time = $(Statistics.median(trial_sc_cuda_gpu_ft32)) on NVIDIA GPU for single column",
)

# multiple column liner interpolation tests without extrapolation
xmin, xmax, nsource, ntarget, nlon, nlat = get_dims_multicol(Float32)
trial_mc_cuda_gpu_ft32 =
    test_multiple_columns(CuArray, Float32, xmin, xmax, nsource, ntarget, nlon, nlat)
println(
    "median time = $(Statistics.median(trial_mc_cuda_gpu_ft32)) on NVIDIA GPU for $nlon x $nlat columns with Float32",
)

xmin, xmax, nsource, ntarget, nlon, nlat = get_dims_multicol(Float64)
trial_mc_cuda_gpu_ft64 =
    test_multiple_columns(CuArray, Float64, xmin, xmax, nsource, ntarget, nlon, nlat)
println(
    "median time = $(Statistics.median(trial_mc_cuda_gpu_ft64)) on NVIDIA GPU for $nlon x $nlat columns with Float64",
)
