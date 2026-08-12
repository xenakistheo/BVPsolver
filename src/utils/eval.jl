function rollout(ctx, θ; dense = false)
    rhs!(du, u, p, t) = (du .= f_theta(ctx, u, t, p); nothing)
    prob = ODEProblem(rhs!, ctx.u0, ctx.tspan, θ)
    kw   = (; verbose = SL.None(), ctx.spec.solve_kwargs...)
    return dense ? solve(prob, ctx.spec.solver; kw...) :
                   solve(prob, ctx.spec.solver; saveat = ctx.tsteps, kw...)
end

function metrics(ctx, θ)
    sol = rollout(ctx, θ)
    if !successful_retcode(sol)
        println("  rollout FAILED: $(sol.retcode) at t=$(sol.t[end]) of $(ctx.tspan[2])")
        return (nrmse = Inf, l2 = Inf)
    end
    P = Array(sol)
    return (nrmse = sqrt(mean(abs2, (P .- ctx.Ydata) ./ ctx.yscale)),
            l2    = sqrt(sum(abs2, P .- ctx.Ydata)))
end

function report(ctx, label, θ)
    m = metrics(ctx, θ)
    println("  [$label]  nrmse=$(round(m.nrmse; sigdigits=4))  l2=$(round(m.l2; sigdigits=4))")
    flush(stdout)
    return m
end

function panel_grid(d)
    d <= 8 && return (1, d, (760, 240 * d))
    ncol = d <= 24 ? 4 : d <= 64 ? 6 : 8
    nrow = cld(d, ncol)
    return (ncol, nrow, (min(1800, 230 * ncol), min(14000, 175 * nrow)))
end

function panel_style(c, d, ncol, nrow)
    big = ncol > 1
    (; title = big ? "$c" : "", titlefontsize = big ? 7 : 11,
       xlabel = (!big || c > d - ncol) ? "t" : "",
       tickfontsize = big ? 5 : 8, legendfontsize = big ? 5 : 8,
       grid = !big, ms = big ? 1.3 : 3, msw = big ? 0.3 : 1.0, lw = big ? 1 : 2,
       left_margin = big ? 2Plots.mm : 0Plots.mm,
       bottom_margin = big ? 1Plots.mm : 0Plots.mm)
end

# train_mask, when given, splits the data scatter into train/test coloring with a
# vertical boundary line at the end of the training window (see extrapolation_ctx).
function plot_fit(ctx, θ; dir = ".", train_mask = nothing)
    logt  = is_log_spaced(ctx.tsteps)
    tgrid = logt ? exp.(range(log(ctx.tsteps[1]), log(ctx.tsteps[end]); length = 400)) :
                   collect(range(ctx.tspan[1], ctx.tspan[2]; length = 400))
    sol  = rollout(ctx, θ; dense = true)
    ok   = successful_retcode(sol)
    ts   = ok ? sort(unique(vcat(tgrid, sol.t))) : Float64[]
    pred = ok ? reduce(hcat, (sol(t) for t in ts)) : zeros(ctx.d, 0)
    ncol, nrow, sz = panel_grid(ctx.d)
    train = train_mask === nothing ? trues(length(ctx.tsteps)) : train_mask
    panels = map(1:ctx.d) do c
        s = panel_style(c, ctx.d, ncol, nrow)
        p = plot(; title = "y$c", titlefontsize = s.titlefontsize,
                 xlabel = s.xlabel, xscale = logt ? :log10 : :identity, legend = c == 1,
                 tickfontsize = s.tickfontsize, legendfontsize = s.legendfontsize,
                 grid = s.grid, left_margin = s.left_margin, bottom_margin = s.bottom_margin)
        scatter!(p, ctx.tsteps[train], ctx.Ydata[c, train];
                 label = c == 1 ? (train_mask === nothing ? "data" : "train data") : "",
                 mc = :white, msc = :black, ms = s.ms, msw = s.msw)
        if train_mask !== nothing
            scatter!(p, ctx.tsteps[.!train], ctx.Ydata[c, .!train]; label = c == 1 ? "test data" : "",
                     mc = :white, msc = :steelblue, ms = s.ms, msw = s.msw)
            vline!(p, [ctx.tsteps[train][end]]; label = c == 1 ? "train/test split" : "",
                   color = :gray, ls = :dash)
        end
        ok && plot!(p, ts, pred[c, :]; label = c == 1 ? "neural ODE" : "",
                    color = :orangered, lw = s.lw)
        p
    end
    suffix = train_mask === nothing ? "fit" : "fit_extrapolated"
    path = joinpath(dir, "$(ctx.spec.name)_$suffix.png")
    savefig(plot(panels...; layout = (nrow, ncol), size = sz), path)
    println("saved $path")
end


function eigenvalues(ctx, θ; absolute=false)
    eigenvalues_true = zeros(ComplexF64, size(ctx.Ydata))
    eigenvalues_learned = zeros(ComplexF64, size(ctx.Ydata))

    for i in 1:size(ctx.Ydata)[2]
        y = ctx.Ydata[:, i]
        t = ctx.tsteps[i]
        F_true(x) = (du = similar(x); ctx.spec.true_ode!(du, x, nothing, t); du)
        F_learned(x) = f_theta(ctx, x, t, θ)
        J_true = ForwardDiff.jacobian(F_true, y)
        eigenvalues_true[:, i] = sort(eigvals(J_true), by = abs, rev = true)
        J_learned = ForwardDiff.jacobian(F_learned, y)
        eigenvalues_learned[:, i] = sort(eigvals(J_learned), by = abs, rev = true)
    end

    if absolute
        eigenvalues_true = abs.(eigenvalues_true)
        eigenvalues_learned = abs.(eigenvalues_learned)
    end

    return eigenvalues_true, eigenvalues_learned
end


# Shared "true vs learned" panel plot for values sampled pointwise at ctx.tsteps
# (used by the spectral and velocity-field fits, which have no continuous ODE
# solution to draw the way plot_fit's dense rollout does). train_mask, when given,
# splits the true-value scatter into train/test coloring with a boundary line.
function plot_pointwise_fit(ctx, Ytrue, Yhat, ylabel, name_suffix; dir = ".", train_mask = nothing)
    logt = is_log_spaced(ctx.tsteps)
    ncol, nrow, sz = panel_grid(ctx.d)
    train = train_mask === nothing ? trues(length(ctx.tsteps)) : train_mask
    panels = map(1:ctx.d) do c
        s = panel_style(c, ctx.d, ncol, nrow)
        p = plot(; title = "$ylabel$c", titlefontsize = s.titlefontsize,
                 xlabel = s.xlabel, xscale = logt ? :log10 : :identity, legend = c == 1,
                 tickfontsize = s.tickfontsize, legendfontsize = s.legendfontsize,
                 grid = s.grid, left_margin = s.left_margin, bottom_margin = s.bottom_margin)
        scatter!(p, ctx.tsteps[train], Ytrue[c, train];
                 label = c == 1 ? (train_mask === nothing ? "true" : "true (train)") : "",
                 mc = :white, msc = :black, ms = s.ms, msw = s.msw)
        if train_mask !== nothing
            scatter!(p, ctx.tsteps[.!train], Ytrue[c, .!train]; label = c == 1 ? "true (test)" : "",
                     mc = :white, msc = :steelblue, ms = s.ms, msw = s.msw)
            vline!(p, [ctx.tsteps[train][end]]; label = c == 1 ? "train/test split" : "",
                   color = :gray, ls = :dash)
        end
        plot!(p, ctx.tsteps, Yhat[c, :]; label = c == 1 ? "learned" : "",
              color = :orangered, lw = s.lw)
        p
    end
    suffix = train_mask === nothing ? name_suffix : "$(name_suffix)_extrapolated"
    path = joinpath(dir, "$(ctx.spec.name)_$suffix.png")
    savefig(plot(panels...; layout = (nrow, ncol), size = sz), path)
    println("saved $path")
end

function plot_spectral_fit(ctx, θ; dir = ".", train_mask = nothing)
    eigenvalues_true, eigenvalues_learned = eigenvalues(ctx, θ; absolute = true)
    plot_pointwise_fit(ctx, eigenvalues_true, eigenvalues_learned, "λ", "spectral_fit";
                        dir, train_mask)
end

function plot_vf_fit(ctx, θ; dir = ".", train_mask = nothing)
    Vtrue = true_vector_field(ctx.spec, ctx.Ydata, ctx.tsteps)
    Vhat  = f_theta(ctx, ctx.Ydata, ctx.tsteps, θ)
    plot_pointwise_fit(ctx, Vtrue, Vhat, "f", "vf_fit"; dir, train_mask)
end

# ── Extrapolation window ─────────────────────────────────────────────────
is_log_spaced(tsteps) = all(>(0), tsteps) && tsteps[end] / max(tsteps[1], eps()) > 100

function extend_tsteps(tsteps, T, T_test, n_extra)
    logt = is_log_spaced(tsteps)
    new  = logt ? exp.(range(log(T), log(T_test); length = n_extra + 1))[2:end] :
                  collect(range(T, T_test; length = n_extra + 1))[2:end]
    return sort(unique(vcat(tsteps, new)))
end

# Extends ctx's training window [0, T] out to an extrapolation window [T, T_test],
# regenerating ground truth on the extended grid. Doubling the window means doubling
# the linear span on a linear time axis, but doubling the number of decades on a log
# axis (e.g. rober), since a flat "2T" there would barely extend past training in
# log-space. Returns the extended ctx plus a boolean mask selecting the training columns.
function extrapolation_ctx(ctx; t_test = nothing, n_extra = nothing)
    spec    = ctx.spec
    T       = spec.tspan[2]
    n_train = length(ctx.tsteps)
    logt    = is_log_spaced(ctx.tsteps)
    T_test  = t_test !== nothing ? t_test : (logt ? T^2 / ctx.tsteps[1] : 2T)
    T_test > T || error("t_test ($T_test) must be greater than the training horizon T=$T")
    n_extra = n_extra === nothing ? 2n_train : n_extra

    tsteps_ext = extend_tsteps(ctx.tsteps, T, T_test, n_extra)
    tspan_ext  = (spec.tspan[1], T_test)
    prob       = ODEProblem(spec.true_ode!, spec.u0, tspan_ext)
    Ydata_ext  = Array(solve(prob, spec.solver; saveat = tsteps_ext, spec.solve_kwargs...))

    ext = merge(ctx, (; tsteps = tsteps_ext, tspan = tspan_ext,
                        Ydata = Ydata_ext, tlen = length(tsteps_ext)))
    return ext, tsteps_ext .<= T
end

# ── Relative error metrics (see metrics.md) ──────────────────────────────
function relative_error(Yhat, Ytrue)
    denom = max.(vec(sum(abs2, Ytrue; dims = 2)), floatmin())
    per_species = vec(sum(abs2, Yhat .- Ytrue; dims = 2)) ./ denom
    return per_species, mean(per_species)
end

function true_vector_field(spec, Ydata, tsteps)
    d, n = size(Ydata)
    V  = similar(Ydata)
    du = zeros(eltype(Ydata), d)
    for i in 1:n
        spec.true_ode!(du, view(Ydata, :, i), nothing, tsteps[i])
        V[:, i] .= du
    end
    return V
end

# Trajectory / velocity-field / spectral relative errors of the model on ctx's time
# grid, each as (per-species values, aggregated mean) — see metrics.md. NaN-filled
# if the rollout fails (e.g. an extrapolation window the model blows up on).
function error_metrics(ctx, θ)
    sol = rollout(ctx, θ)
    if !successful_retcode(sol)
        nanvec = fill(NaN, ctx.d)
        return (traj_species = nanvec, traj = NaN, vf_species = nanvec, vf = NaN,
                spec_species = nanvec, spec = NaN)
    end
    Yhat = Array(sol)
    traj_species, traj = relative_error(Yhat, ctx.Ydata)

    Vhat  = f_theta(ctx, ctx.Ydata, ctx.tsteps, θ)
    Vtrue = true_vector_field(ctx.spec, ctx.Ydata, ctx.tsteps)
    vf_species, vf = relative_error(Vhat, Vtrue)

    eig_true, eig_learned = eigenvalues(ctx, θ; absolute = true)
    spec_species, spec_err = relative_error(eig_learned, eig_true)

    return (; traj_species, traj, vf_species, vf, spec_species, spec = spec_err)
end
