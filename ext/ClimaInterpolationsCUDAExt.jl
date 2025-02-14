module ClimaInterpolationsCUDAExt

using CUDA

import ClimaInterpolations
import ClimaInterpolations.Interpolation1D:
    interpolate,
    interpolate1d!,
    Order1D,
    Extrapolate1D,
    Linear,
    Flat,
    LinearExtrapolation,
    get_stencil

include(joinpath("cuda", "interpolation1d.jl"))
include(joinpath("cuda", "bilinearinterpolation.jl"))
end # module
