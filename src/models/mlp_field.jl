gelu(x) = 0.5 * x * (1 + tanh(0.7978845608028654 * (x + 0.044715 * x^3)))

struct MLPField{TR,H} <: Lux.AbstractLuxContainerLayer{(:trunk, :head)}
    trunk::TR
    head::H
    tmid::Float64
    tscale::Float64
    kappa::Vector{Float64}
    fscale::Vector{Float64}
    time_dependent::Bool
end

init_stiff!(ps, ::MLPField) = ps

function (m::MLPField)(u::AbstractVecOrMat, t, ps, st)
    feats = m.time_dependent ? vcat(u, time_feature(m, u, t)) : u
    h, _ = m.trunk(feats, ps.trunk, st.trunk)
    y, _ = m.head(h, ps.head, st.head)
    return m.fscale .* y, st
end

function build_mlp_field(d, Ydata, tsteps, yscale; width, depth,
                         scaling = :none, activation = tanh, time_dependent = false,
                         init = :default)
    slope = max.(vec(maximum(abs.(diff(Ydata, dims = 2) ./ diff(tsteps)'), dims = 2)), 1e-30)
    tmid   = Float64((first(tsteps) + last(tsteps)) / 2)
    tscale = max(Float64((last(tsteps) - first(tsteps)) / 2), eps(Float64))
    fs = scaling === :eq ? collect(Float64, yscale) ./ (2 * tscale) : ones(d)
    nin = d + (time_dependent ? 1 : 0)
    kw = init_kwargs(init)
    trunk = Chain(Dense(nin, width, activation; kw...),
                  (Dense(width, width, activation; kw...) for _ in 2:depth)...)
    return MLPField(trunk, Dense(width, d; kw...), tmid, tscale,
                    KAPPA_FRAC .* slope, fs, time_dependent)
end
