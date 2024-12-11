import ClimaInterpolations.Interpolation1D:
    interpolate, Linear, get_stencil, interpolate1d!, Flat, LinearExtrapolation
using Test


function get_uniform_column_grids(
    ::Type{FT},
    xmin,
    xmax,
    xmintarg,
    xmaxtarg,
    nsource,
    ntarget,
) where {FT}
    return (
        Vector{FT}(range(xmin, xmax, length = nsource)),
        Vector{FT}(range(xmintarg, xmaxtarg, length = ntarget)),
    )
end

function test_single_column(
    ::Type{FT},
    xmin,
    xmax,
    nsource,
    ntarget;
    toler,
    xmintarg = xmin,
    xmaxtarg = xmax,
    extrapolation = Flat(),
) where {FT}
    @testset "1D linear interpolation on single column with $FT" begin
        xsource, xtarget =
            get_uniform_column_grids(FT, xmin, xmax, xmintarg, xmaxtarg, nsource, ntarget)
        fsource = sin.(xsource) # function defined on source grid
        ftarget = zeros(FT, ntarget) # allocated function on target grid
        interpolate1d!(ftarget, xsource, xtarget, fsource, Linear(), extrapolation)
        diff = maximum(
            abs.(ftarget .- sin.(xtarget)) .* (xtarget .≤ xmax) .* (xtarget .≥ xmin),
        )
        @test diff ≤ toler
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

    end
    return nothing
end

function test_multiple_columns(
    ::Type{FT},
    xmin,
    xmax,
    nsource,
    ntarget,
    nlon,
    nlat;
    toler,
    xmintarg = xmin,
    xmaxtarg = xmax,
    extrapolation = Flat(),
) where {FT}
    @testset "1D linear interpolation on multiple columns with $FT" begin
        xsource, xtarget =
            get_uniform_column_grids(FT, xmin, xmax, xmintarg, xmaxtarg, nsource, ntarget)

        xsourcecols = repeat(xsource, 1, nlon, nlat)
        xtargetcols = repeat(xtarget, 1, nlon, nlat)
        fsourcecols = sin.(xsourcecols)
        ftargetcols = zeros(FT, ntarget, nlon, nlat)

        interpolate1d!(
            ftargetcols,
            xsourcecols,
            xtargetcols,
            fsourcecols,
            Linear(),
            extrapolation,
        )
        diff = maximum(abs.(ftargetcols .- sin.(xtargetcols))[:])
        @test diff ≤ toler
    end
    return nothing
end

# single column linear interpolation tests without extrapolation
test_single_column(Float32, 0.0, 2 * π, 150, 200, toler = 0.0003)
test_single_column(Float64, 0.0, 2 * π, 150, 200, toler = 0.0003)
# single column linear interpolation tests with Flat extrapolation
test_single_column(
    Float32,
    0.0,
    2 * π,
    150,
    200,
    toler = 0.0003,
    xmintarg = -1.0,
    xmaxtarg = 2 * π + 1.0,
    extrapolation = Flat(),
)
test_single_column(
    Float64,
    0.0,
    2 * π,
    150,
    200,
    toler = 0.0003,
    xmintarg = -1.0,
    xmaxtarg = 2 * π + 1.0,
    extrapolation = Flat(),
)
# multiple column liner interpolation tests without extrapolation
test_multiple_columns(Float32, 0.0, 2 * π, 150, 200, 2560, 1280, toler = 0.0003)
test_multiple_columns(Float64, 0.0, 2 * π, 150, 200, 2560, 1280, toler = 0.0003)
