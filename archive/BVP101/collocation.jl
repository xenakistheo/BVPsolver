# using Optimization
# using OptimizationOptimJL
# using ADTypes
# using ForwardDiff
using Plots
using BenchmarkTools
using OrdinaryDiffEq
using NonlinearSolve


function F!(u, p, t)
    """u0 = 1"""
    return -u 
end



function G!(u, p, t)
    """u0 = 1"""
    return -p*(u - cos(t)) - sin(t)
end 


const RadauIIa3_points = ((4-sqrt(6))/10, (4+sqrt(6))/10, 1)


function collocation_step(f, yn, h, tn, c; p=nothing, x0=nothing)
    """
    f(t, y, p)
    """
    s = length(c)
    
    #TODO: Optimize using Horner's method
    u(x, t) = yn + sum(x[i]*(t-tn)^i for i in 1:s)
    uprime(x, t) = sum(i*x[i]*(t-tn)^(i-1) for i in 1:s)
    
    l(x, p, i) = (uprime(x, tn + c[i]*h) - f(u(x, tn + c[i]*h), p, tn + c[i]*h))^2
    loss_fn(x, p) = sum(l(x, p, i) for i in 1:s)

    residual!(r, x, p) = for i in 1:s
        r[i] = uprime(x, tn + c[i]*h) - f(u(x, tn + c[i]*h), p, tn + c[i]*h)
    end 

    if x0 != nothing
        @assert length(x0) == s
    else 
        x0 = yn*ones(s)
    end 

    # optf = OptimizationFunction(loss_fn, AutoForwardDiff())
    # prob = OptimizationProblem(optf, x0, p)
    # sol = solve(prob, LBFGS())
    prob = NonlinearProblem(residual!, x0, p)

    return solve(prob, NewtonRaphson()).u
end

collocation_step(F!, 1, 0.1, 0, RadauIIa3_points)

function collocation_solve(f, y0, tspan, dt; c=RadauIIa3_points, p=nothing)
    s = length(c)
    T = range(tspan[1], tspan[2]; step = dt)
    Y = similar(T)
    Y[1] = y0
    yn = y0 
    coeffs = zeros(s)
    x0 = zeros(s)

    for (i, tn) in enumerate(T[1:end-1])
        coeffs .= collocation_step(f, yn, dt, tn, c; x0=x0, p=p)
        x0 .= coeffs
        Y[i+1] = yn + sum(coeffs[k]*dt^k for k in 1:s)
        yn = Y[i+1]
    end 
    return T, Y
end 
    
tspan = (0, 20)
u0 = 1

λ = 100_000
dt = 0.00002 # For λ = 100.000, RK4 and Tsit5 - both cannot go up one factor in stepsize 
# dt = 0.000002 # For λ = 1.000.000 # Breaks RAM
# dt = 0.000000002 # For λ = 10.000.000# No convergence. Breaks RAM. 


prob = ODEProblem(G!, u0, tspan, λ)
@benchmark solve(prob, RK4(); dt=dt, adaptive=false) 
#Median: 96.051 ms for for λ = 100.000
sol = solve(prob, RK4(); dt=dt, adaptive=false)
trk4, yrk4 = sol.t, sol.u

@benchmark collocation_solve(G!, u0, tspan, 0.2; p=λ) 
# Median: 84.712ms for λ = 100.000, dt = 0.00002
# Median: 111.275ms for λ = 1.000.000
# Median: 122.611ms for λ = 10.000.000
tc, yc = collocation_solve(G!, u0, tspan, 0.2; p=λ)

prob = ODEProblem(G!, u0, tspan, λ)
@benchmark solve(prob, Tsit5(); dt=dt, adaptive=false)
#Median: 142.537 ms for for λ = 100.000, dt = 0.00002
sol = solve(prob, Tsit5(); dt=dt, adaptive=false)
ttsit5, ytsit5 = sol.t, sol.u

plot(tc, yc, label="Collocation")
plot!(ttsit5, ytsit5, label="Tsit5")
plot!(trk4, yrk4, label="RK4", linestyle=:dash)

"""
As we can see for the examples above, for (super)-stiff problems, 
    the collocation method is just much better. It seems almost completely 
    unaffected by the stiffness of the problem. Yes, a bit more compute
    overhead. However, one should also consider that I did not optimise it 
    one bit. 

Update: Changed from formalizing the collocation as an optimisation problem, 
to instead formulating it as finding the roots of a nonlinear function. 
    I.e. instead of using L-BFGS, I am using NewtonRaphson
this speeded up collocation_solve from 84.712ms to 5.267 ms !!!


- Comment: It is wrong to say that collocation method is independent of stiffness. 
 the NewtonRaphson method is probably the bottleneck. 

- Comment: RadauII is more than the collocation points. It has a butcher tableau. 
    One can use this tableau to solve (directly) for the stage derivatives Ki instead of 
    the polynomical coefficients. This is probably faster, although my implementation
    is nicer for pedagogical purposes. 

        
        
#TODO

- Comment, it would be nice to say something about the "eigenvalues" of the ODE
        and why the difference of them is what makes it stiff. 
        - and exactly what we want to capture with our stiffened net. 
        Investigate this

- Comment: Extend to BVPs not only IVP

- Play around with differnet collocation points. 
"""