using Lux, Random

struct StiffLayer{F,I} <: Lux.AbstractLuxLayer
    sigma::F
    width::Int
    init::I
end

StiffLayer(sigma::F, width::Integer; init=(rng, m) -> randn(Float32, m)) where {F} =
    StiffLayer{F,typeof(init)}(sigma, width, init)

function Lux.initialparameters(rng::AbstractRNG, l::StiffLayer)
    vals = l.init isa Function ? l.init(rng, l.width) : l.init
    @assert length(vals) == l.width "Init vector length $(length(vals)) must equal width=$(l.width)"
    return (ps = collect(vals),)
end

Lux.initialstates(::AbstractRNG, ::StiffLayer) = NamedTuple()

function Lux.apply(l::StiffLayer, x, ps, st)
    xmat = ndims(x) == 1 ? reshape(x, :, 1) : x
    width, _ = size(xmat)
    psv = ps.ps
    @assert length(psv) >= width "Need at least $width parameters, got $(length(psv))"
    scales = @inbounds [l.sigma(psv, j) for j in 1:width]
    y = xmat .* reshape(scales, width, 1)
    return (ndims(x) == 1 ? vec(y) : y), st
end

struct StiffLayer2{F,I} <: Lux.AbstractLuxLayer
    sigma::F
    nparams::Int
    init::I
end

StiffLayer2(sigma::F; P::Integer, init=(rng, p) -> randn(Float32, p)) where {F} =
    StiffLayer2{F,typeof(init)}(sigma, P, init)

function Lux.initialparameters(rng::AbstractRNG, l::StiffLayer2)
    vals = l.init isa Function ? l.init(rng, l.nparams) : l.init
    @assert length(vals) == l.nparams "Init vector length $(length(vals)) must equal P=$(l.nparams)"
    return (ps = collect(vals),)
end

Lux.initialstates(::AbstractRNG, ::StiffLayer2) = NamedTuple()

function Lux.apply(l::StiffLayer2, x, ps, st)
    xmat = ndims(x) == 1 ? reshape(x, :, 1) : x
    width, _ = size(xmat)
    psv = ps.ps
    @assert length(psv) == l.nparams "Parameter length $(length(psv)) must equal P=$(l.nparams)"
    scales = @inbounds [l.sigma(psv, j) for j in 1:width]
    y = xmat .* reshape(scales, width, 1)
    return (ndims(x) == 1 ? vec(y) : y), st
end


##########
# First Idea
sigmoid01(u::T) where {T<:Real} = one(T) / (one(T) + exp(-u))

struct StateDepStiffLayer1{G,T} <: Lux.AbstractLuxLayer
    gate::G
    m::Int
    bmin::T
    bmax::T
end

function StateDepStiffLayer1(gate, m::Integer; bmin=0.0f0, bmax=5.0f0)
    return StateDepStiffLayer1(gate, m, bmin, bmax)
end

function Lux.initialparameters(rng::AbstractRNG, l::StateDepStiffLayer1)
    return (
        gate = Lux.initialparameters(rng, l.gate),
    )
end

function Lux.initialstates(rng::AbstractRNG, l::StateDepStiffLayer1)
    return (
        gate = Lux.initialstates(rng, l.gate),
        idx = Float32.(1:l.m),
    )
end

function (l::StateDepStiffLayer1)(input, ps, st)
    y, x = input

    ymat = ndims(y) == 1 ? reshape(y, :, 1) : y
    xmat = ndims(x) == 1 ? reshape(x, :, 1) : x

    width, batch = size(ymat)
    @assert width == l.m "Expected y to have width $(l.m), got $width"

    raw, st_gate = l.gate(xmat, ps.gate, st.gate)

    @assert size(raw, 1) == 2 "gate network must output 2 values: a(x), raw_b(x)"

    a = raw[1:1, :]
    raw_b = raw[2:2, :]

    # bounded positive b(x)
    b = l.bmin .+ (l.bmax - l.bmin) .* sigmoid01.(raw_b)

    idx = reshape(st.idx, l.m, 1)

    u = a .+ idx .* b

    T = eltype(u)
    scales = one(T) ./ (one(T) .+ exp.(u))

    z = ymat .* scales

    new_st = (
        gate = st_gate,
        idx = st.idx,
    )

    return ndims(y) == 1 ? vec(z) : z, new_st
end


#############
# Second idea
struct StateDepStiffLayer2{G,T} <: Lux.AbstractLuxLayer
    gate::G
    m::Int
    amin::T
    amax::T
    bmin::T
    bmax::T
end

function StateDepStiffLayer2(
    gate,
    m::Integer;
    amin=0.0f0,
    amax=5.0f0,
    bmin=0.0f0,
    bmax=5.0f0,
)
    return StateDepStiffLayer2(gate, m, amin, amax, bmin, bmax)
end

function Lux.initialparameters(rng::AbstractRNG, l::StateDepStiffLayer2)
    return (
        gate = Lux.initialparameters(rng, l.gate),
    )
end

function Lux.initialstates(rng::AbstractRNG, l::StateDepStiffLayer2)
    return (
        gate = Lux.initialstates(rng, l.gate),
        idx = Float32.(1:l.m),
    )
end

function (l::StateDepStiffLayer2)(input, ps, st)
    y, x = input

    ymat = ndims(y) == 1 ? reshape(y, :, 1) : y
    xmat = ndims(x) == 1 ? reshape(x, :, 1) : x

    width, batch = size(ymat)
    @assert width == l.m "Expected y to have width $(l.m), got $width"

    raw, st_gate = l.gate(xmat, ps.gate, st.gate)

    @assert size(raw, 1) == 2 "gate network must output 2 values: raw_a(x), raw_b(x)"

    raw_a = raw[1:1, :]
    raw_b = raw[2:2, :]

    a = l.amin .+ (l.amax - l.amin) .* sigmoid01.(raw_a)
    b = l.bmin .+ (l.bmax - l.bmin) .* sigmoid01.(raw_b)

    idx = reshape(st.idx, l.m, 1)

    scales = exp.(-a .- idx .* b)

    z = ymat .* scales

    new_st = (
        gate = st_gate,
        idx = st.idx,
    )

    return ndims(y) == 1 ? vec(z) : z, new_st
end

#####
# Wrapper 
struct StiffenedNet{V,S,U} <: Lux.AbstractLuxLayer
    V::V
    S::S
    U::U
end

function Lux.initialparameters(rng::AbstractRNG, l::StiffenedNet)
    return (
        V = Lux.initialparameters(rng, l.V),
        S = Lux.initialparameters(rng, l.S),
        U = Lux.initialparameters(rng, l.U),
    )
end

function Lux.initialstates(rng::AbstractRNG, l::StiffenedNet)
    return (
        V = Lux.initialstates(rng, l.V),
        S = Lux.initialstates(rng, l.S),
        U = Lux.initialstates(rng, l.U),
    )
end

function (l::StiffenedNet)(x, ps, st)
    y, st_V = l.V(x, ps.V, st.V)

    z, st_S = l.S((y, x), ps.S, st.S)

    out, st_U = l.U(z, ps.U, st.U)

    new_st = (
        V = st_V,
        S = st_S,
        U = st_U,
    )

    return out, new_st
end

# ##### Example usage
# input_dim = 2
# m = 16
# output_dim = 2

# V = Lux.Chain(
#     Lux.Dense(input_dim => 32, tanh),
#     Lux.Dense(32 => m),
# )

# U = Lux.Chain(
#     Lux.Dense(m => 32, tanh),
#     Lux.Dense(32 => output_dim),
# )

# gate = Lux.Chain(
#     Lux.Dense(input_dim => 16, tanh),
#     Lux.Dense(16 => 2),
# )


# # For idea 1
# S1 = StateDepStiffLayer1(
#     gate,
#     m;
#     bmin=0.0f0,
#     bmax=5.0f0,
# )

# model1 = StiffenedNet(V, S1, U)

# ## For idea 2
# S2 = StateDepStiffLayer2(
#     gate,
#     m;
#     amin=0.0f0,
#     amax=3.0f0,
#     bmin=0.0f0,
#     bmax=5.0f0,
# )

# model2 = StiffenedNet(V, S2, U)