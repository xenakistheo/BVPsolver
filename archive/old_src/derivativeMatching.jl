"""
This script implements the collocation approach based on derivative matching from Roesch, 2021. 

It is meant as a fast initial training, to get the parameters in the right ballpark, before switching over to some more accurate and expensive method
such as collocation with a BVP solver, or shooting. 
"""

using DiffEqFlux
using Printf
using Zygote

"""Copied from StiffNN/collocation/collocationSpiral.jl, with some modifications"""
function train!(
    node,
    dudt_model,
    ps,
    st,
    u0,
    ode_data,
    tsteps;
    nepochs::Int = 300,
    learning_rate::Float32 = 0.04f0,
    min_learning_rate::Float32 = 1f-4,
    lr_schedule::Symbol = :cosine,          # :constant or :cosine
    loss_mode::Symbol = :collocation,       # :collocation, :standard, :hybrid
    lambda_traj::Float32 = 0.1f0,
    hybrid_every::Int = 1,                  # only used when loss_mode == :hybrid
    print_every::Int = 10,
    plot_every::Int = 50
)


    # ---------------------------
    # Prediction function
    # ---------------------------
    function predict_neuralode(p)
        Array(node(u0, p, st)[1])
    end

    # ---------------------------
    # Collocation targets
    # ---------------------------
    U_hat_prime, U_hat = collocate_data(ode_data, tsteps, EpanechnikovKernel())

    # ---------------------------
    # Loss definitions
    # ---------------------------
    @views function collocation_loss(p)
        loss = zero(eltype(U_hat))
        for i in axes(U_hat_prime, 2)
            d = U_hat_prime[:, i] - dudt_model(U_hat[:, i], p, st)[1]
            loss += sum(abs2, d)
        end
        return loss
    end

    function trajectory_loss(p)
        pred = predict_neuralode(p)
        loss = zero(eltype(pred))
        @inbounds @simd for i in eachindex(pred, ode_data)
            d = pred[i] - ode_data[i]
            loss += d * d
        end
        return loss
    end

    function total_loss(p, epoch)
        if loss_mode == :collocation
            return collocation_loss(p)

        elseif loss_mode == :standard
            return trajectory_loss(p)

        elseif loss_mode == :hybrid
            loss_col = collocation_loss(p)

            # Only include trajectory term every `hybrid_every` epochs
            if epoch % hybrid_every == 0
                loss_traj = trajectory_loss(p)
                return loss_col + lambda_traj * loss_traj
            else
                return loss_col
            end

        else
            error("Unknown loss_mode = $loss_mode. Use :collocation, :standard, or :hybrid")
        end
    end

    # ---------------------------
    # Learning-rate scheduler
    # ---------------------------
    function current_lr(epoch)
        if lr_schedule == :constant
            return learning_rate

        elseif lr_schedule == :cosine
            # cosine decay from learning_rate to min_learning_rate
            progress = (epoch - 1) / max(nepochs - 1, 1)
            cosine_factor = 0.5f0 * (1f0 + cos(Float32(pi) * progress))
            return min_learning_rate + (learning_rate - min_learning_rate) * cosine_factor

        else
            error("Unknown lr_schedule = $lr_schedule. Use :constant or :cosine")
        end
    end

    # ---------------------------
    # Optimizer state
    # ---------------------------
    opt = Optimisers.Adam(learning_rate)
    opt_state = Optimisers.setup(opt, ps)

    losses = Vector{Float32}(undef, nepochs)
    lrs = Vector{Float32}(undef, nepochs)

    # ---------------------------
    # Training loop
    # ---------------------------
    for epoch in 1:nepochs

        # Update learning rate by rebuilding Adam with the new LR
        lr_epoch = current_lr(epoch)
        opt = Optimisers.Adam(lr_epoch)

        # 1) Compute loss and gradients
        loss_val, back = Zygote.pullback(p -> total_loss(p, epoch), ps)
        grads = back(one(loss_val))[1]

        # 2) Optimizer step
        opt_state, ps = Optimisers.update(opt_state, ps, grads, opt)

        losses[epoch] = Float32(loss_val)
        lrs[epoch] = lr_epoch

        # Print progress
        if epoch == 1 || epoch % print_every == 0 || epoch == nepochs
            @printf "Epoch %4d | Loss = %.6f | LR = %.6e\n" epoch loss_val lr_epoch
        end

    end

    println("\n✓ Training complete!")
    return ps, st, losses, lrs
end