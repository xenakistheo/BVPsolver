using OrdinaryDiffEq
using OrdinaryDiffEqTsit5: Tsit5
using OrdinaryDiffEqRosenbrock: Rodas5, Rodas5P
using OrdinaryDiffEqSDIRK: Kvaerno5

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

function make_problem(name::AbstractString; T::Type{<:AbstractFloat}=Float32, profile::Symbol=:default,
                      mu::Real=100.0, epsilon::Real=0.01, n::Integer=8)
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
        kwargs = fast ? (; abstol=T(1e-10), reltol=T(1e-10)) : (; abstol=T(1e-10), reltol=T(1e-10))
        return ProblemSpec{T,typeof(Rodas5()),typeof(true_ode!)}(
            "rober", u0, tspan, tsteps, Rodas5(), kwargs, true_ode!
        )
    end

    if lname == "vanderpol"
        u0 = T[2.0, 0.0]
        tspan = fast ? (T(0.0), T(200.0)) : (T(0.0), T(400.0))
        μ = T(mu)
        true_ode! = function (du, u, p, t)
            du[1] = u[2]
            du[2] = μ * (1 - u[1]^2) * u[2] - u[1]
        end
        kwargs = fast ? (; abstol=T(1e-7), reltol=T(1e-7)) : (; abstol=T(1e-8), reltol=T(1e-8))
        tsteps = solve(ODEProblem(true_ode!, u0, tspan), Kvaerno5(); kwargs...).t
        return ProblemSpec{T,typeof(Kvaerno5()),typeof(true_ode!)}(
            "vanderpol", u0, tspan, tsteps, Kvaerno5(), kwargs, true_ode!
        )
    end

    if lname == "pollu"
        u0 = zeros(T, 20)
        u0[2], u0[4], u0[7], u0[8], u0[9], u0[17] = T(0.2), T(0.04), T(0.1), T(0.3), T(0.01), T(0.007)
        tspan = (T(0.0), T(60.0))
        tlen = fast ? 50 : 100
        k = T[0.35, 26.6, 1.23e4, 8.6e-4, 8.2e-4, 1.5e4, 1.3e-4, 2.4e4, 1.65e4, 9.0e3,
              2.2e-2, 1.2e4, 1.88, 1.63e4, 4.8e7, 3.5e-4, 1.75e-2, 1.0e9, 4.44e12,
              1.24e4, 2.1, 5.78, 4.74e-2, 1.78e3, 3.12]
        true_ode! = function (dy, y, p, t)
            r1  = k[1] * y[1];        r2  = k[2] * y[2] * y[4];   r3  = k[3] * y[5] * y[2]
            r4  = k[4] * y[7];        r5  = k[5] * y[7];          r6  = k[6] * y[7] * y[6]
            r7  = k[7] * y[9];        r8  = k[8] * y[9] * y[6];   r9  = k[9] * y[11] * y[2]
            r10 = k[10] * y[11] * y[1]; r11 = k[11] * y[13];      r12 = k[12] * y[10] * y[2]
            r13 = k[13] * y[14];      r14 = k[14] * y[1] * y[6];  r15 = k[15] * y[3]
            r16 = k[16] * y[4];       r17 = k[17] * y[4];         r18 = k[18] * y[16]
            r19 = k[19] * y[16];      r20 = k[20] * y[17] * y[6]; r21 = k[21] * y[19]
            r22 = k[22] * y[19];      r23 = k[23] * y[1] * y[4];  r24 = k[24] * y[19] * y[1]
            r25 = k[25] * y[20]
            dy[1]  = -r1 - r10 - r14 - r23 - r24 + r2 + r3 + r9 + r11 + r12 + r22 + r25
            dy[2]  = -r2 - r3 - r9 - r12 + r1 + r21
            dy[3]  = -r15 + r1 + r17 + r19 + r22
            dy[4]  = -r2 - r16 - r17 - r23 + r15
            dy[5]  = -r3 + r4 + r4 + r6 + r7 + r13 + r20
            dy[6]  = -r6 - r8 - r14 - r20 + r3 + r18 + r18
            dy[7]  = -r4 - r5 - r6 + r13
            dy[8]  = r4 + r5 + r6 + r7
            dy[9]  = -r7 - r8
            dy[10] = -r12 + r7 + r9
            dy[11] = -r9 - r10 + r8 + r11
            dy[12] = r9
            dy[13] = -r11 + r10
            dy[14] = -r13 + r12
            dy[15] = r14
            dy[16] = -r18 - r19 + r16
            dy[17] = -r20
            dy[18] = r20
            dy[19] = -r21 - r22 - r24 + r23 + r25
            dy[20] = -r25 + r24
        end
        kwargs = (; abstol=T(1e-25), reltol=T(1e-10))
        tgrid = vcat(T(0), T(10) .^ range(log10(T(1e-5)), log10(tspan[2]); length = tlen - 1))
        tsteps = sort(unique(tgrid))
        return ProblemSpec{T,typeof(Rodas5()),typeof(true_ode!)}(
            "pollu", u0, tspan, tsteps, Rodas5(), kwargs, true_ode!
        )
    end

    if lname == "hires"
        u0 = zeros(T, 8)
        u0[1], u0[8] = T(1.0), T(0.0057)
        tspan = (T(0.0), T(321.8122))
        tlen = fast ? 50 : 100
        true_ode! = function (du, u, p, t)
            y1, y2, y3, y4, y5, y6, y7, y8 = u
            du[1] = -T(1.71) * y1 + T(0.43) * y2 + T(8.32) * y3 + T(0.0007)
            du[2] = T(1.71) * y1 - T(8.75) * y2
            du[3] = -T(10.03) * y3 + T(0.43) * y4 + T(0.035) * y5
            du[4] = T(8.32) * y2 + T(1.71) * y3 - T(1.12) * y4
            du[5] = -T(1.745) * y5 + T(0.43) * y6 + T(0.43) * y7
            du[6] = -T(280.0) * y6 * y8 + T(0.69) * y4 + T(1.71) * y5 - T(0.43) * y6 + T(0.69) * y7
            du[7] = T(280.0) * y6 * y8 - T(1.81) * y7
            du[8] = -T(280.0) * y6 * y8 + T(1.81) * y7
        end
        kwargs = fast ? (; abstol=T(1e-10), reltol=T(1e-8)) : (; abstol=T(1e-14), reltol=T(1e-10))
        tgrid = vcat(T(0), min.(T(10) .^ range(log10(T(1e-3)), log10(tspan[2]); length=tlen - 1), tspan[2]))
        tsteps = sort(unique(tgrid))
        return ProblemSpec{T,typeof(Rodas5()),typeof(true_ode!)}(
            "hires", u0, tspan, tsteps, Rodas5(), kwargs, true_ode!
        )
    end

    if lname == "orego"
        u0 = T[1.0, 2.0, 3.0]
        tspan = fast ? (T(0.0), T(180.0)) : (T(0.0), T(360.0))
        s, q, w = T(77.27), T(8.375e-6), T(0.161)
        true_ode! = function (du, u, p, t)
            x, y, z = u
            du[1] = s * (y + x * (1 - q * x - y))
            du[2] = (z - (1 + x) * y) / s
            du[3] = w * (x - z)
        end
        kwargs = fast ? (; abstol=T(1e-10), reltol=T(1e-8)) : (; abstol=T(1e-14), reltol=T(1e-10))
        place = (; abstol=T(1e-3), reltol=T(1e-3))
        tsteps = solve(ODEProblem(true_ode!, u0, tspan), Rodas5(); place...).t
        return ProblemSpec{T,typeof(Rodas5()),typeof(true_ode!)}(
            "orego", u0, tspan, tsteps, Rodas5(), kwargs, true_ode!
        )
    end

    if lname == "davis-skodje"
        u0 = T[1.0, 2.0]
        tspan = (T(0.0), T(15.0))
        tlen = fast ? 300 : 600
        γ = T(1 / epsilon)
        true_ode! = function (du, u, p, t)
            x, y = u
            du[1] = -x
            du[2] = -γ * y + ((γ - 1) * x + γ * x^2) / (1 + x)^2
        end
        kwargs = (; abstol=T(1e-10), reltol=T(1e-8))
        tgrid = vcat(T(0), min.(T(10) .^ range(log10(T(1e-4)), log10(tspan[2]); length=tlen - 1), tspan[2]))
        tsteps = sort(unique(tgrid))
        return ProblemSpec{T,typeof(Rodas5P()),typeof(true_ode!)}(
            "davis-skodje", u0, tspan, tsteps, Rodas5P(), kwargs, true_ode!
        )
    end


    if lname == "brusselator"
        Nb = n
        xyd = collect(range(T(0), stop = T(1), length = Nb))
        dx = xyd[2] - xyd[1]
        A, B, alpha = T(3.4), T(1.0), T(10.0)
        adx2 = alpha / dx^2
        U0 = zeros(T, Nb, Nb, 2)
        for i in 1:Nb, j in 1:Nb
            U0[i, j, 1] = T(22) * (xyd[j] * (1 - xyd[j]))^T(1.5)
            U0[i, j, 2] = T(27) * (xyd[i] * (1 - xyd[i]))^T(1.5)
        end
        u0 = vec(U0)
        tspan = (T(0.0), T(11.5))
        lim(a) = a == Nb + 1 ? 1 : a == 0 ? Nb : a
        true_ode! = function (du, u, p, t)
            U  = reshape(u, Nb, Nb, 2)
            dU = reshape(du, Nb, Nb, 2)
            on = t >= T(1.1)
            @inbounds for j in 1:Nb, i in 1:Nb
                ip1, im1, jp1, jm1 = lim(i + 1), lim(i - 1), lim(j + 1), lim(j - 1)
                x, y = xyd[i], xyd[j]
                src = (on && ((x - T(0.3))^2 + (y - T(0.6))^2 <= T(0.1)^2)) ? T(5.0) : zero(T)
                u1, u2 = U[i, j, 1], U[i, j, 2]
                dU[i, j, 1] = adx2 * (U[im1, j, 1] + U[ip1, j, 1] + U[i, jp1, 1] +
                                      U[i, jm1, 1] - 4u1) +
                              B + u1^2 * u2 - (A + 1) * u1 + src
                dU[i, j, 2] = adx2 * (U[im1, j, 2] + U[ip1, j, 2] + U[i, jp1, 2] +
                                      U[i, jm1, 2] - 4u2) +
                              A * u1 - u1^2 * u2
            end
        end
        kwargs = fast ? (; abstol=T(1e-10), reltol=T(1e-8), tstops=[T(1.1)]) :
                        (; abstol=T(1e-14), reltol=T(1e-10), tstops=[T(1.1)])
        place = (; abstol=T(1e-6), reltol=T(1e-6), tstops=[T(1.1)])
        tsteps = solve(ODEProblem(true_ode!, u0, tspan), Rodas5(); place...).t
        return ProblemSpec{T,typeof(Rodas5()),typeof(true_ode!)}(
            "brusselator", u0, tspan, tsteps, Rodas5(), kwargs, true_ode!
        )
    end

    error("Unknown problem '$name'. Valid: spiral, rober, vanderpol, pollu, hires, orego, davis-skodje, brusselator")
end
