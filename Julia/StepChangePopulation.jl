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


function densityFlatPop(τ, c, s)
    pushfirst!(s, 0)

    n = length(s)

    density = 1

    for i in 2:n
        density *= binomial(n-i+2,2)* exp(-binomial(n-i+2,2)*(s[i] - s[i-1]))
    end 
    
    return density 
end

function densityFlatPop2(τ, c, s)\
    pushfirst!(s, 0)

    n = length(s)

    logSummands = [binomial(n-j+2,2) for j in 2:n]
    logSummands = log.(logSummands)

    intSummands = zeros(n-1)

    for j in 2:n
        intSummands[j-1] = (s[j]-s[j-1])
        intSummands[j-1] *= binomial(n-j+2,2)
    end

    total = sum(logSummands - intSummands)
    return exp(total)
end
function densityStepChange(τ, c, s)
    
    # τ is the step change point
    # c is the intensity after τ
    # s is the time vector

    pushfirst!(s, 0)
    n = length(s)
    stepPoint = binSearch(τ, s)

    logSummands = [binomial(n-j+2,2)*(1-(1-c)*(j>=stepPoint)) for j in 2:n]
    logSummands = log.(logSummands)

    intSummands = zeros(n-1)
    
    
    for j in 2:n
        intSummands[j-1] = (s[j]-s[j-1])*((j < stepPoint) + c*(j-1>=stepPoint)) + (τ-s[j-1] + c*(s[j]-τ))*(j == stepPoint)
        intSummands[j-1] *= binomial(n-j+2,2)
    end

    total = sum(logSummands - intSummands)
    return exp(total)
end

#function for change of variables so that we can integrate on a hypercube
function changeVar(func, τ, c, s)
    # τ is the step change point
    # c is the intensity after τ
    # s = (s_1, s_2, ..., s_{n-1}, t_n) is a vector such that t_i = t_n * ∏_{j=i}^{n-1} s_j
    n = length(s) 
    times = [s[end]*prod(s[j] for j in i:n-1) for i in 1:n-1]
    push!(times, s[end])

    jacobian = prod([s[j]^(j-1) for j in eachindex(s)])

    return func(τ, c, times)*jacobian
end

fStepChange(u, p) = changeVar(densityStepChange, p[1], p[2], u)

fConst(u,p) = changeVar(densityFlatPop, p[1], p[2], u)


function solveInt(func, n, τ, c, reltol)
    lower_terminals = zeros(n)
    upper_terminals = push!(ones(n-1), Inf)
    domain = (lower_terminals, upper_terminals)

    problem = IntegralProblem(func, domain, [τ, c])

    sol= solve(problem, HCubatureJL(); reltol=reltol)

    return sol.u
end

