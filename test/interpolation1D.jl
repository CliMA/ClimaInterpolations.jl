import ClimaInterpolations.Interpolation1D:
    Linear, interpolate1d!, Flat, LinearExtrapolation

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
) where {DA, FT}
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
            abs.(ftarget .- sin.(xtarget)) .* (xtarget .≤ xmax) .*
            (xtarget .≥ xmin),
        )
        @test diff ≤ toler
        converttoarray = !(DA <: Array)
        xtarget = converttoarray ? Array(xtarget) : xtarget
        ftarget = converttoarray ? Array(ftarget) : ftarget
        fsource = converttoarray ? Array(fsource) : fsource
        # test extrapolation
        if xmintarg < xmin || xmaxtarg > xmax
            if extrapolation == Flat()
                left_boundary_pass = true
                right_boundary_pass = true
                for i in 1:length(xtarget)
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
) where {DA, FT}
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
    end
    return nothing
end

get_dims_singlecol(::Type{FT}) where {FT} = (FT(0), FT(2π), 150, 200)
get_dims_multicol(::Type{FT}) where {FT} = (FT(0), FT(2π), 150, 200, 1280, 640)
