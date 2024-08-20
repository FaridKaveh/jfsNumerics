# don't need this but adding for future compatibity, in case binomial moves from base
using Combinatorics 
using Integrals
using Cubature


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


function densityStepChange(s::AbstractVector{Float64}, p)
    #the probability density evaluated at s when there is a step change at p[1] down/up to p[2] 
    τ = p[1]
    c = p[2]
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

    
    return exp(sum(logSummands .- intSummands))
end

function branchLengthRatios(s::AbstractVector{Float64}, i, j) 
    #This is the ratio S_i*S_j/(T_{tot}^\ast)^2
    i, j = Int(i), Int(j)
    n= length(s)
    s[n-i+1]*s[n-j+1]/(sum(s)+s[n])^2
end 
Integrand(u, p) = densityStepChange(u, p[1:2])*branchLengthRatios(u, p[3], p[4])


#function for change of variables so that we can integrate on a hypercube
function changeVar(func, s, p)
    # τ is the step change point
    # c is the intensity after τ
    # s = (s_1, s_2, ..., s_{n-1}, t_n) is a vector such that t_i = t_n * ∏_{j=i}^{n-1} s_j
    n = length(s) 
    times = [s[end]*prod(s[j] for j in i:n-1) for i in 1:n-1]
    push!(times, s[end])

    jacobian = prod([s[j]^(j-1) for j in eachindex(s)])

    return func(times, p)*jacobian
end


IntegrandOnHypercube(u, p) = changeVar(Integrand, u, p)

fStepChange(u, p) = changeVar(densityStepChange, u, p)


function fStepChange_InPlace(y, u, p)
    y[1] = changeVar(densityStepChange, u, p) 
end 

function fBranchLengthRatios(y, u::AbstractVector{Float64}, p)
    ## The last two paramers of p are the indices i and j
    y[1] = changeVar(Integrand, u, p) 
end

function solveIntOOP(func, n, p, reltol)
    #OOP is Out of Place
    lower_terminals = zeros(n)
    upper_terminals = push!(ones(n-1), Inf)
    domain = (lower_terminals, upper_terminals)

    problem = IntegralProblem(func, domain, p)

    sol= solve(problem, CubatureJLh(); reltol=reltol)

    return sol.u
end

function solveIntIP(func, n, p, reltol)

    lower_terminals = zeros(n)
    upper_terminals = push!(ones(n-1), Inf)
    domain = (lower_terminals, upper_terminals)

    prototype = zeros(1)

    problem = IntegralProblem(IntegralFunction(func, prototype), domain, p)

    sol= solve(problem, VEGAS(); reltol=reltol)

    return sol.u
end 

function makeCorrelMatIP(n, τ, c, reltol)
    correlMat = zeros(n-1, n-1)
    for i in 1:n-1
        for j in 1:i
            correlMat[i,j] = solveIntIP(IntegrandOnHypercube, n-1, [τ, c, i, j], reltol)[1]
            i != j ? correlMat[j,i] = correlMat[i,j] : nothing
        end
    end

    return correlMat
end

function makeProbMat(n, τ, c, reltol)
    correlMat = makeCorrelMatIP(n, τ, c, reltol)

    probMat = zeros(n-1, n-1)

    for i in 1:n-1
        for j in 1:i

            if i == n-1
                probMat[i,j] = j == n-1 ? n^2 * correlMat[i,j] : n*(j+1)*(correlMat[i,j]-correlMat[i,j+1])
            else 
                probMat[i,j] = (i+1)*(j+1)*(correlMat[i,j]-correlMat[i+1,j]+correlMat[i+1,j+1]-correlMat[i,j+1])
            end 
            
            i != j ? probMat[j,i] = probMat[i,j] : nothing
        end 
    end 

    return probMat
end

ProbMat = makeProbMat(5, 1, 1 , 1e-6)

