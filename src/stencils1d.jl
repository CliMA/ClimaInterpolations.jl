
abstract type Order1D end
abstract type Extrapolate1D end

struct Linear <: Order1D end

struct Flat <: Extrapolate1D end
struct LinearExtrapolation <: Extrapolate1D end

"""
    get_stencil(alg::Linear, xsource, xtarget; first = 1, extrapolate = Flat())

Get the interval in the source grid (`xsource`) in which the target point, `xtarget`, is located.
For linear interpolation, this interval is also the full stencil needed for the interpolation.
The stencil specification is characterized by `[st, en]` where `st` is the starting point and
`en` is the ending point of the stencil in the target grid. This function returns the tuple `(st, en)`.
Linear and Flat extrapolation schemes are supported at the boundaries. The argument `first` can be used
to speedup the search, by providing a more efficient starting point for the search.
"""
function get_stencil(alg::Linear, xsource, xtarget; first = 1, extrapolate = Flat())
    last = length(xsource)
    if xtarget < xsource[1] # extrapolation
        # Linear extrapolation
        extrapolate == LinearExtrapolation() && return (1, 2)
        # Flat extrapolation by default
        return (1, 1)
    end
    if xtarget > xsource[last] # extrapolation
        # Linear extrapolation
        extrapolate == LinearExtrapolation() && return (last - 1, last)
        # Flat extrapolation by default
        return (last, last)
    end
    st = first
    for i = first:last
        if xtarget ≤ xsource[i]
            st = max(i - 1, 1)
            break
        end
    end
    return (st, st < last ? st + 1 : st)
end
