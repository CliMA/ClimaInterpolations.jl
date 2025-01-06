module ClimaInterpolationsMetalExt

using Metal

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

include(joinpath("metal", "interpolation1d.jl"))
include(joinpath("metal", "bilinearinterpolation.jl"))
end
