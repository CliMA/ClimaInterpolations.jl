
abstract type Order1D end
abstract type Extrapolate1D end

struct Linear <: Order1D end

struct Flat <: Extrapolate1D end
struct LinearExtrapolation <: Extrapolate1D end

function get_stencil(alg::Linear, xsource, xtarg; first = 1, extrapolate = Flat())
    last = length(xsource)
    if xtarg < xsource[1] # extrapolation
        # Linear extrapolation
        extrapolate == LinearExtrapolation() && return (1, 2)
        # Flat extrapolation by default
        return (1, 1)
    end
    if xtarg > xsource[last] # extrapolation
        # Linear extrapolation
        extrapolate == LinearExtrapolation() && return (last - 1, last)
        # Flat extrapolation by default
        return (last, last)
    end
    st = first
    for i = first:last
        if xtarg ≤ xsource[i]
            st = max(i - 1, 1)
            break
        end
    end
    return (st, st < last ? st + 1 : st)
end
