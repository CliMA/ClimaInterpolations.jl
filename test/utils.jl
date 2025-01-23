function get_uniform_column_grids(
    ::Type{DA},
    ::Type{FT},
    xmin,
    xmax,
    xmintarg,
    xmaxtarg,
    nsource,
    ntarget,
) where {DA, FT}
    return (
        DA{FT}(range(xmin, xmax, length = nsource)),
        DA{FT}(range(xmintarg, xmaxtarg, length = ntarget)),
    )
end
