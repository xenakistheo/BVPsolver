
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
    chi::Vector{Float64}
    kappa::Vector{Float64}
    lam::Vector{Float64}
    gbound::Vector{Float64}
    abound::Float64
    signed_loss::Bool
end

init_stiff!(ps, m::StiffProductionLossField) =
    (ps.logloss.bias .= m.signed_loss ? -1.0 : -6.0; ps)

function (m::StiffProductionLossField)(u::AbstractVecOrMat, ps, st)
    x̂ = (u .- m.ymid) ./ m.yscale
    z = asinh.(u ./ m.chi) ./ ZNORM
    h, _  = m.trunk(vcat(x̂, z), ps.trunk, st.trunk)
    pa, _ = m.prod(h, ps.prod, st.prod)
    lb, _ = m.logloss(h, ps.logloss, st.logloss)
    f = m.kappa .* sinh.(m.abound .* tanh.(pa ./ m.abound))
    c = m.gbound .* tanh.(lb ./ m.gbound)
    return (m.signed_loss ? f .+ m.lam .* sinh.(c) .* u
                          : f .- m.lam .* exp.(c) .* u), st
end

function build_stiff_field(d, Ydata, tsteps, ymid, yscale; width, depth, signed_loss)
    slope = max.(vec(maximum(abs.(diff(Ydata, dims = 2) ./ diff(tsteps)'), dims = 2)), 1e-30)
    umax  = max.(vec(maximum(abs.(Ydata), dims = 2)), 1e-30)
    lam   = slope ./ max.(collect(Float64, yscale), 1e-12 .* umax, floatmin())
    rmax  = 10 / minimum(diff(tsteps))
    trunk = Chain(Dense(2d, width, tanh), (Dense(width, width, tanh) for _ in 2:depth)...)
    return StiffProductionLossField(
        trunk, Dense(width, d), Dense(width, d),
        collect(Float64, ymid), collect(Float64, yscale),
        CHI_FRAC .* umax, KAPPA_FRAC .* slope, lam,
        clamp.(asinh.(rmax ./ lam), 1.0, asinh(1e6)),
        asinh(PROD_HEADROOM), signed_loss)
end

f_theta(ctx, u, θ) = ctx.model(u, ComponentVector(θ, ctx.ps_axes), ctx.st)[1]
