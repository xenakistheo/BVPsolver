# Show how to extract parameters from NN - edit - and put back in

using Lux
using Random
using ComponentArrays


eltype = Float32

rng = Xoshiro()

model = Chain(
    Dense(2, 4, tanh),
    Dense(4, 4, tanh),
    Dense(4, 2)
)

ps, st = Lux.setup(rng, model)

ps_ca = ComponentVector(ps)
ps_vec = collect(ps_ca)

ps_axes = getaxes(ps_ca)

new_ps_vec = 2 .* ps_vec
ps_reconstructed = ComponentVector(new_ps_vec, ps_axes)


model([1f0,1f0], ps, st)[1]