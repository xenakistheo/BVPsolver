using Optimization
using OptimizationOptimJL
using ADTypes

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p = [1.0, 100.0]

prob = OptimizationProblem(rosenbrock, x0, p)

using OptimizationOptimJL
sol = solve(prob, NelderMead())




function F!(t, u, p)
    """u0 = 1"""
    return -u 
end



function G!(t, u, p)
    """u0 = 1"""
    return -1000*(u - cos(t)) - sin(t)
end 


const RadauIIa3_points = ((4-sqrt(6))/10, (4+sqrt(6))/10, 1)


function collocation_step(f, yn, h, tn, c; p=nothing)
    """
    f(t, y, p)
    """
    s = length(c)
    
    u(x, t) = yn + sum([x[i]*t^i for i in 1:s])
    uprime(x, t) = sum([i*x[i]*t^(i-1) for i in 1:s])
    
    l(x, p, i) = (uprime(x, tn + c[i]*h) - f(tn + c[i]*h, u(x, tn + c[i]*h), p))^2
    loss_fn(x, p) = sum([l(x, p, i) for i in 1:s])

    x0 = yn*ones(s)
    optf = OptimizationFunction(loss_fn, AutoForwardDiff())
    prob = OptimizationProblem(optf, x0, p)
    sol = solve(prob, LBFGS())

    return sol
end

collocation_step(F!, 1, 0.1, 0, c)

function collocation_solve(f, y0, tspan, dt; c=RadauIIa3_points)
    T = range(tspan[1], tspan[2]; step = dt)
    Y = similar(T)
    Y[1] = y0
    yn = y0 

    for (i, t) in enumerate(tspan[1:end-1])
        coeffs = collocation_step(f, yn, dt, tn, c)
        Y[i+1] = yn +  sum([coeffs[k]*(t+dt)^k for k in 1:s])
        yn = Y[i+1]
    end 
    return T, Y
end 
    

tc, yc = collocation_solve(F!, 1, (0, 1), 0.01)

plot(tc, yc)

