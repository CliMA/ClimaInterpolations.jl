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
import ClimaInterpolations.BilinearInterpolation:
    Bilinear, set_source_range!, interpolatebilinear!, get_stencil_bilinear1d, get_dims

include(joinpath("metal", "interpolation1d.jl"))
include(joinpath("metal", "bilinearinterpolation.jl"))

end
