glorot_uniform(rng, dims::Integer...) =
    (rand(rng, Float32, dims...) .- 0.5f0) .* Float32(sqrt(24 / (dims[1] + dims[end])))

zeros_init(rng, dims::Integer...) = zeros(Float32, dims...)

init_kwargs(init) = init === :glorot ?
    (; init_weight = glorot_uniform, init_bias = zeros_init) : (;)
