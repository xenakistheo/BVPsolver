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
