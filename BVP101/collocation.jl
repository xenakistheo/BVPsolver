using Optimization
using OptimizationOptimJL
using ADTypes
using ForwardDiff
using Plots
using BenchmarkTools
using OrdinaryDiffEq


function F!(u, p, t)
    """u0 = 1"""
    return -u 
end



function G!(u, p, t)
    """u0 = 1"""
    return -100000*(u - cos(t)) - sin(t)
end 


const RadauIIa3_points = ((4-sqrt(6))/10, (4+sqrt(6))/10, 1)


function collocation_step(f, yn, h, tn, c; p=nothing)
    """
    f(t, y, p)
    """
    s = length(c)
    
    u(x, t) = yn + sum([x[i]*(t-tn)^i for i in 1:s])
    uprime(x, t) = sum([i*x[i]*(t-tn)^(i-1) for i in 1:s])
    
    l(x, p, i) = (uprime(x, tn + c[i]*h) - f(u(x, tn + c[i]*h), p, tn + c[i]*h))^2
    loss_fn(x, p) = sum([l(x, p, i) for i in 1:s])

    x0 = yn*ones(s)
    optf = OptimizationFunction(loss_fn, AutoForwardDiff())
    prob = OptimizationProblem(optf, x0, p)
    sol = solve(prob, LBFGS())

    return sol.u
end

collocation_step(F!, 1, 0.1, 0, RadauIIa3_points)

function collocation_solve(f, y0, tspan, dt; c=RadauIIa3_points)
    s = length(c)
    T = range(tspan[1], tspan[2]; step = dt)
    Y = similar(T)
    Y[1] = y0
    yn = y0 
    coeffs = zeros(s)

    for (i, tn) in enumerate(T[1:end-1])
        coeffs .= collocation_step(f, yn, dt, tn, c)
        Y[i+1] = yn + sum([coeffs[k]*dt^k for k in 1:s])
        yn = Y[i+1]
    end 
    return T, Y
end 
    
tspan = (0, 20)
dt = 0.00002 # For λ = 100.000
# dt = 0.000002 # For λ = 1.000.000 # Breaks RAM
# dt = 0.000000002 # For λ = 10.000.000# No convergence. Breaks RAM. 


prob = ODEProblem(G!, 1, tspan)
@benchmark solve(prob, RK4(); dt=dt, adaptive=false) 
#Median: 96.051 ms for for λ = 100.000
sol = solve(prob, RK4(); dt=dt, adaptive=false)
trk4, yrk4 = sol.t, sol.u

@benchmark collocation_solve(G!, 1, tspan, 0.2) 
# Median: 84.712ms for λ = 100.000,  
# Median: 111.275ms for λ = 1.000.000
# Median: 122.611ms for λ = 10.000.000
tc, yc = collocation_solve(G!, 1, tspan, 0.2)


plot(tc, yc, label="Collocation")
plot!(trk4, yrk4, label="RK4", linestyle=:dash)

"""
As we can see for the examples above, for (super)-stiff problems, 
    the collocation method is just much better. It seems almost completely 
    unaffected by the stiffness of the problem. Yes, a bit more compute
    overhead. However, one should also consider that I did not optimise it 
    one bit. 

"""