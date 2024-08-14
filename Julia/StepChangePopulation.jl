# don't need this but adding for future compatibity, in case binomial moves from base
using Combinatorics 
using Integrals


function binSearch(x, ls)
    # binary search for x in ls
    # returns the index of the first element in ls that is greater than x
    # if x is greater than all elements in ls, returns length(ls) + 1
    # if x is less than all elements in ls, returns 1

    low = 1
    high = length(ls)

    while low <= high
        mid = (low + high) ÷ 2

        if ls[mid] > x
            high = mid - 1
        else
            low = mid + 1
        end
    end

    return low
end

function logDensityStepChange(τ, c, s)
    
    # τ is the step change point
    # c is the intensity after τ
    # s is the time vector

    pushfirst!(s, 0)
    n = length(s)
    stepPoint = binSearch(τ, s)

    logSummands = log.([binomial(n-j+2,2)*(1-(1-c)*(j>=stepPoint)) for j in 2:n])

    intSummands = zeros(n-1)
    
    
    for j in 1:n-1
        intSummands[j] = (s[j+1]-s[j])*((j+1 < stepPoint) + c*(j>stepPoint)) + (τ-s[j] + c*(s[j+1]-τ))*(j+1 == stepPoint)
        intSummands[j] *= binomial(n-j+1,2)
    end

    return sum(logSummands) - sum(intSummands)
end

#function for change of variables so that we can integrate on a hypercube
function changeVar(τ, c, s)
    # τ is the step change point
    # c is the intensity after τ
    # s = (s_1, s_2, ..., s_{n-1}, t_n) is a vector such that t_i = t_n * ∏_{j=i}^{n-1} s_j
    n = length(s) 
    times = [s[end]*prod(s[j] for j in i:n-1) for i in 1:n-1]
    push!(times, s[end])

    return logDensityStepChange(τ, c, times)
end

#The post factor is the jacobian of the transformation
f(u, p) = exp(changeVar(p[1], p[2], u))*prod([u[j]^(j-1) for j in eachindex(u)])
    

function solvef(n, τ, c, reltol)
    lower_terminals = zeros(n)
    upper_terminals = push!(ones(n), Inf)
    domain = (lower_terminals, upper_terminals)

    problem = IntegralProblem(f, domain, [τ, c])

    sol= solve(problem, HCubatureJL(); reltol=reltol)

    return sol.u
end


solvef(4, 1, 1, 1e-6)