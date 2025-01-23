# ClimaInterpolations.jl

## Interpolation1D
`interpolate1d!` function can be used to perform linear interpolation on a single column on or a collection of columns. Both single threaded CPU and NVIDIA GPU platforms are supported. Examples for single column and multiple column cases are presented below.

### Example

Example for interpolating a single column on a CPU:

```julia
import ClimaInterpolations.Interpolation1D:
    Linear, interpolate1d!, Flat

FT, DA = Float32, Array
xmin, xmax, nsource, ntarget = FT(0), FT(2π), 150, 200
xmintarg, xmaxtarg = xmin, xmax

xsource = DA{FT}(range(xmin, xmax, length = nsource))
xtarget = DA{FT}(range(xmintarg, xmaxtarg, length = ntarget))

fsource = sin.(xsource) # function defined on source grid
ftarget = DA(zeros(FT, ntarget)) # allocated function on target grid
interpolate1d!(ftarget, xsource, xtarget, fsource, Linear(), Flat())
```

Example for interpolating multiple columns on a CPU:

```julia
import ClimaInterpolations.Interpolation1D:
    Linear, interpolate1d!, Flat

FT, DA = Float32, Array
xmin, xmax, nsource, ntarget, nlat, nlon = FT(0), FT(2π), 150, 200, 1280, 640
xmintarg, xmaxtarg = xmin, xmax

xsource = DA{FT}(range(xmin, xmax, length = nsource))
xtarget = DA{FT}(range(xmintarg, xmaxtarg, length = ntarget))

xsourcecols = DA(repeat(xsource, 1, nlon, nlat))
xtargetcols = DA(repeat(xtarget, 1, nlon, nlat))
fsourcecols = DA(sin.(xsourcecols))
ftargetcols = DA(zeros(FT, ntarget, nlon, nlat))

interpolate1d!(ftargetcols, xsourcecols, xtargetcols, fsourcecols, Linear(), Flat())
```

The above examples can be run on NVIDIA GPUs by setting `DA = CuArray`.
