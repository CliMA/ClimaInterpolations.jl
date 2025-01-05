function interpolatebilinear!(
    ftarget::MtlArray{FT,N},
    bilinear::B,
    fsource::MtlArray{FT,N},
) where {FT<:Float32,N,B}
    @assert N ≥ 2
    (leveldimssource..., nxs, nys) = size(fsource)
    (leveldimstarget..., nxt, nyt) = size(ftarget)

    nsourcex, nsourcey, ntargetx, ntargety = get_dims(bilinear)

    @assert leveldimssource == leveldimstarget
    @assert nxs == nsourcex && nys == nsourcey
    @assert nxt == ntargetx && nyt == ntargety
    # max threadgroups per grid is not officially specified by Metal
    # It is possible this number is unlimited on newer Apple models (>m3)
    maxxthreadgroups = 1024
    maxythreadgroups = 1024
    # target n loops per dimension per threadgroup
    targetloopsperblockdimx = 4
    targetloopsperblockdimy = 4
    nxthreadgroups = min(cld(ntargetx, targetloopsperblockdimx), maxxthreadgroups)
    nythreadgroups = min(cld(ntargety, targetloopsperblockdimy), maxythreadgroups)

    nxloops = cld(ntargetx, nxthreadgroups)
    nyloops = cld(ntargety, nythreadgroups)

    levelcidxs = CartesianIndices(leveldimssource)
    nlevels = length(levelcidxs)

    kernel = @metal launch = false interpolatebilinear_kernel!(
        ftarget,
        bilinear,
        fsource,
        levelcidxs,
        (nxloops, ntargetx),
        (nyloops, ntargety),
        (1, 1),
    )
    nthreads = min(kernel.pipeline.maxTotalThreadsPerThreadgroup, nlevels)
    nzloops = cld(nlevels, nthreads)

    @metal threads = nthreads groups = (nxthreadgroups, nythreadgroups) interpolatebilinear_kernel!(
        ftarget,
        bilinear,
        fsource,
        levelcidxs,
        (nxloops, ntargetx),
        (nyloops, ntargety),
        (nzloops, nlevels),
    )
    return nothing
end

function interpolatebilinear_kernel!(
    ftarget::AbstractArray{FT,N},
    bilinear::B,
    fsource::AbstractArray{FT,N},
    levelcidxs,
    (nxloops, ntargetx),
    (nyloops, ntargety),
    (nzloops, nlevels),
) where {FT<:Float32,N,B}
    (; sourcex, sourcey, targetx, targety, startx, starty) = bilinear

    ixtg, iytg = threadgroup_position_in_grid_2d()
    izt = thread_index_in_threadgroup()
    nzthreads = threads_per_threadgroup_1d()
    (nxthreadgroups, nythreadgroups) = threadgroups_per_grid_2d()

    @inbounds begin
        for yl = 1:nyloops
            iy = iytg + (yl - 1) * nythreadgroups
            if iy ≤ ntargety
                y, sty = targety[iy], starty[iy]
                y1, y2 = sourcey[sty], sourcey[sty+1]
                dy1 = y - y1
                dy2 = y2 - y
                for xl = 1:nxloops
                    ix = ixtg + (xl - 1) * nxthreadgroups
                    if ix ≤ ntargetx
                        x, stx = targetx[ix], startx[ix]
                        x1, x2 = sourcex[stx], sourcex[stx+1]

                        dx1 = x - x1
                        dx2 = x2 - x

                        fac = FT(1) / ((x2 - x1) * (y2 - y1))

                        for zl = 1:nzloops
                            iz = izt + (zl - 1) * nzthreads
                            if iz ≤ nlevels
                                levelidx = Tuple(levelcidxs[iz])

                                ftarget[levelidx..., ix, iy] =
                                    (
                                        dx1 * (
                                            dy2 * fsource[levelidx..., stx+1, sty] +
                                            dy1 * fsource[levelidx..., stx+1, sty+1]
                                        ) +
                                        dx2 * (
                                            dy1 * fsource[levelidx..., stx, sty+1] +
                                            dy2 * fsource[levelidx..., stx, sty]
                                        )
                                    ) * fac
                            end
                        end
                    end
                end
            end
        end

    end
    return nothing
end

Metal.Adapt.@adapt_structure Bilinear

function Bilinear(sourcex::V, sourcey::V, targetx::V, targety::V) where {V<:MtlVector}
    startx = similar(targetx, Int)
    starty = similar(targety, Int)
    # Call set_source_range_kernel! with one thread per group
    # and two threadgroups
    @metal threads = 1 groups = 2 set_source_range_kernel!(
        startx,
        sourcex,
        targetx,
        starty,
        sourcey,
        targety,
    )
    return Bilinear(sourcex, sourcey, targetx, targety, startx, starty)
end

function set_source_range_kernel!(
    startx::AbstractVector{I},
    sourcex::AbstractVector{FT},
    targetx::AbstractVector{FT},
    starty::AbstractVector{I},
    sourcey::AbstractVector{FT},
    targety::AbstractVector{FT},
) where {I,FT}
    order = Linear()
    groupid = threadgroup_position_in_grid_1d()
    localtid = thread_position_in_threadgroup_1d()
    (groupid == 1 && localtid == 1) && set_source_range!(startx, sourcex, targetx)
    (groupid == 2 && localtid == 1) && set_source_range!(starty, sourcey, targety)
    return nothing
end
