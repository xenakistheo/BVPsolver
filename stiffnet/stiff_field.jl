using Lux, ComponentArrays

const KAPPA_FRAC    = 1e-3
const PROD_HEADROOM = 3e5
const CHI_FRAC      = 1e-3
const ZNORM         = asinh(1 / CHI_FRAC)

struct StiffProductionLossField{TR,P,L} <: Lux.AbstractLuxContainerLayer{(:trunk, :prod, :logloss)}
    trunk::TR
    prod::P
    logloss::L
    ymid::Vector{Float64}
    yscale::Vector{Float64}
    tmid::Float64
    tscale::Float64
    chi::Vector{Float64}
    kappa::Vector{Float64}
    lam::Vector{Float64}
    gbound::Vector{Float64}
    abound::Float64
    signed_loss::Bool
    time_dependent::Bool
end

init_stiff!(ps, m::StiffProductionLossField) =
    (ps.logloss.bias .= m.signed_loss ? -1.0 : -6.0; ps)

function time_feature(m::StiffProductionLossField, u, t)
    if u isa AbstractVector
        τ = t isa Number ? t : only(t)
        return [(τ - m.tmid) / m.tscale]
    end
    n = size(u, 2)
    if t isa Number
        return fill((t - m.tmid) / m.tscale, 1, n)
    end
    length(t) == n || throw(DimensionMismatch(
        "Received $(length(t)) times for $n state columns"))
    return reshape((t .- m.tmid) ./ m.tscale, 1, n)
end

function (m::StiffProductionLossField)(u::AbstractVecOrMat, t, ps, st)
    x̂ = (u .- m.ymid) ./ m.yscale
    z = asinh.(u ./ m.chi) ./ ZNORM
    feats = m.time_dependent ? vcat(x̂, z, time_feature(m, u, t)) : vcat(x̂, z)
    h, _  = m.trunk(feats, ps.trunk, st.trunk)
    pa, _ = m.prod(h, ps.prod, st.prod)
    lb, _ = m.logloss(h, ps.logloss, st.logloss)
    f = m.kappa .* sinh.(m.abound .* tanh.(pa ./ m.abound))
    c = m.gbound .* tanh.(lb ./ m.gbound)
    return (m.signed_loss ? f .+ m.lam .* sinh.(c) .* u
                          : f .- m.lam .* exp.(c) .* u), st
end

function build_stiff_field(d, Ydata, tsteps, ymid, yscale; width, depth, signed_loss,
                           time_dependent = false)
    slope = max.(vec(maximum(abs.(diff(Ydata, dims = 2) ./ diff(tsteps)'), dims = 2)), 1e-30)
    umax  = max.(vec(maximum(abs.(Ydata), dims = 2)), 1e-30)
    lam   = slope ./ max.(collect(Float64, yscale), 1e-12 .* umax, floatmin())
    rmax  = 10 / minimum(diff(tsteps))
    tmid   = Float64((first(tsteps) + last(tsteps)) / 2)
    tscale = max(Float64((last(tsteps) - first(tsteps)) / 2), eps(Float64))
    nin = 2d + (time_dependent ? 1 : 0)
    trunk = Chain(Dense(nin, width, tanh), (Dense(width, width, tanh) for _ in 2:depth)...)
    return StiffProductionLossField(
        trunk, Dense(width, d), Dense(width, d),
        collect(Float64, ymid), collect(Float64, yscale),
        tmid, tscale,
        CHI_FRAC .* umax, KAPPA_FRAC .* slope, lam,
        clamp.(asinh.(rmax ./ lam), 1.0, asinh(1e6)),
        asinh(PROD_HEADROOM), signed_loss, time_dependent)
end

f_theta(ctx, u, t, θ) = ctx.model(u, t, ComponentVector(θ, ctx.ps_axes), ctx.st)[1]
