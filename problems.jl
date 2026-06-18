using OrdinaryDiffEq

struct ProblemSpec{T,S,F}
    name::String
    u0::Vector{T}
    tspan::Tuple{T,T}
    tsteps::Vector{T}
    solver::S
    solve_kwargs::NamedTuple
    true_ode!::F
end

function generate_training_data(spec::ProblemSpec)
    prob = ODEProblem(spec.true_ode!, spec.u0, spec.tspan)
    return Array(solve(prob, spec.solver; saveat=spec.tsteps, spec.solve_kwargs...))
end

function make_problem(name::AbstractString; T::Type{<:AbstractFloat}=Float32, profile::Symbol=:default)
    lname = lowercase(name)
    fast = profile == :fast
    profile ∈ (:default, :fast) || error("Unknown profile '$profile'. Valid: :default, :fast")

    if lname == "spiral"
        u0 = T[2.0, 0.0]
        tspan = (T(0.0), T(1.5))
        tlen = fast ? 20 : 30
        tsteps = collect(range(tspan[1], tspan[2]; length=tlen))
        true_ode! = function (du, u, p, t)
            a = T[-0.1 2.0; -2.0 -0.1]
            du .= ((u .^ 3)' * a)'
        end
        return ProblemSpec{T,typeof(Tsit5()),typeof(true_ode!)}(
            "spiral", u0, tspan, tsteps, Tsit5(), (;), true_ode!
        )
    end

    if lname == "rober"
        u0 = T[1.0, 0.0, 0.0]
        tspan = fast ? (T(1e-6), T(1e3)) : (T(1e-6), T(1e5))
        tlen = fast ? 50 : 100
        tsteps = collect(T(10.0) .^ range(log10(tspan[1]), log10(tspan[2]); length=tlen))
        k1, k2, k3 = T(0.04), T(3e7), T(1e4)
        true_ode! = function (du, u, p, t)
            y1, y2, y3 = u
            du[1] = -k1 * y1 + k3 * y2 * y3
            du[2] = k1 * y1 - k2 * y2^2 - k3 * y2 * y3
            du[3] = k2 * y2^2
        end
        kwargs = fast ? (; abstol=T(1e-6), reltol=T(1e-6)) : (; abstol=T(1e-10), reltol=T(1e-10))
        return ProblemSpec{T,typeof(Rodas5()),typeof(true_ode!)}(
            "rober", u0, tspan, tsteps, Rodas5(), kwargs, true_ode!
        )
    end

    if lname == "vanderpol"
        u0 = T[2.0, 0.0]
        tspan = fast ? (T(0.0), T(200.0)) : (T(0.0), T(400.0))
        tlen = fast ? 250 : 1000
        tsteps = collect(range(tspan[1], tspan[2]; length=tlen))
        μ = T(100.0)
        true_ode! = function (du, u, p, t)
            du[1] = u[2]
            du[2] = μ * (1 - u[1]^2) * u[2] - u[1]
        end
        return ProblemSpec{T,typeof(Kvaerno5()),typeof(true_ode!)}(
            "vanderpol", u0, tspan, tsteps, Kvaerno5(), (;), true_ode!
        )
    end

    error("Unknown problem '$name'. Valid: spiral, rober, vanderpol")
end
