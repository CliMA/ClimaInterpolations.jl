#=
using Revise; using TestEnv; TestEnv.activate(); include("test/runtests.jl")
=#
using Test
using ClimaInterpolations
using SafeTestsets
using Aqua

@safetestset "interpolation1d" begin
    include("interpolation1D.jl")

    for FT in (Float32, Float64)
        # single column linear interpolation tests without extrapolation
        test_single_column(Array, FT, get_dims_singlecol(FT)...)

        # single column linear interpolation tests with Flat extrapolation
        xmin, xmax, nsource, ntarget = get_dims_singlecol(FT)
        test_single_column(
            Array,
            FT,
            xmin,
            xmax,
            nsource,
            ntarget,
            xmintarg = xmin - 1.0,
            xmaxtarg = xmax + 1.0,
            extrapolation = Flat(),
        )
        # multi-column linear interpolation tests without extrapolation
        xmin, xmax, nsource, ntarget, nlon, nlat = get_dims_multicol(FT)
        test_multiple_columns(
            Array,
            FT,
            xmin,
            xmax,
            nsource,
            ntarget,
            nlon,
            nlat,
        )
    end
end

#! format: off
@safetestset "Aqua" begin @time include("aqua.jl") end
#! format: on
