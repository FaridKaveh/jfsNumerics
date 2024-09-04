# don't need this but adding for future compatibity, in case binomial moves from base
using Combinatorics 
using Integrals
using Cubature
using Cuba
using MonteCarloIntegration



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

function densityExponentialGrowth(s, β)
    #the probability density evaluated at s when there is exponential growth with rate β
    # meaning that at time t in the past the population was e^{-βt} smaller than at present
    # β is the growth rate
    # s is the time vector

    pushfirst!(s, 0)
    n = length(s)

    binomialCoeff = [binomial(j, 2) for j = 2:n] 
    logSummands = log.(binomialCoeff) .+ β*s[2:end]
    
    expTimes = exp.(β * s)
    intSummands = expTimes[2:end] - expTimes[1:end-1]
    intSummands = intSummands .* binomialCoeff

    # if minimum([intSummands; logSummands]) < 1e-12 || maximum([intSummands; logSummands]) > 1e12
    #     println("intSummands =")
    #     println(intSummands)
    #     println("logSummands")
    #     println(logSummands)
    #     println("return val=")
    #     println(exp(sum(logSummands .- intSummands)))
    #     println("------------------------------------------------")
    # end 

    exponent = sum(intSummands .- logSummands)

    

    return exponent ≤ 1e16 ? exp(-exponent) : 0.0
    
end 


function densityStepChange(s, p)
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
    
    
    for j = 2:n
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

function VectorIntegrand(u, p)
    # integrand evaluation for all pairs (i,j) in the correlation matrix
    # p[1] = τ, p[2] = c

    n= length(u)
    y = zeros(Int((n*(n-1)/2+n)))
    for i = 1:n
        k = i >= 2 ? sum(1:i-1) : 0
        for j = 1:i
            y[k+j] = branchLengthRatios(u, i, j)
        end
    end 
    return densityStepChange(u, p[1:2])*y
end 

function VectorIntegrandExp(u, p)
    # integrand evaluation for all pairs (i,j) in the correlation matrix
    # p[1] = β

    n= length(u)
    y = zeros(Int((n*(n-1)/2+n)))
    for i = 1:n
        k = i >= 2 ? sum(1:i-1) : 0
        for j = 1:i
            y[k+j] = branchLengthRatios(u, i, j)
        end
    end 
    return densityExponentialGrowth(u, p[1])*y
end 

#function for change of variables so that we can integrate on a hypercube
function changeVar(func, s, p)
    
    # s = (s_1, s_2, ..., s_{n-1}, t_n) is a vector such that t_i = t_n * ∏_{j=i}^{n-1} s_j
    n = length(s) 
    times = [s[end]*prod(s[j] for j in i:n-1) for i in 1:n-1]
    push!(times, s[end])

    jacobian = prod([s[j]^(j-1) for j in eachindex(s)])

    return func(times, p)*jacobian
end

function changeVarHyperCube(func, s, p)

    #s = (s_1, s_2, ..., s_{n-1}, u) is a vector such that t_n = u/(1-u) and t_i = t_n * ∏_{j=i}^{n-1} s_j
    # this change of variable takes the unit hypercube to {0 < t_1 < t_2 < ... < t_{n-1} < t_n < ∞}
    n = length(s)
    tn = s[end]/(1-s[end])
    times = [tn*prod(s[j] for j in i:n-1) for i in 1:n-1]
    push!(times, tn)

    jacobian = prod([s[j]^(j-1) for j in 1:n-1]) * tn^(n-1) * 1/(1-s[end])^2
    
    return func(times, p) *jacobian
end

IntegrandSemiInfinite(u, p) = changeVar(Integrand, u, p)

IntegrandHyperCube(u, p) = changeVarHyperCube(Integrand, u, p)

VectorIntegrandHyperCube(u,p) = changeVarHyperCube(VectorIntegrand, u, p)

StepChangeIntegrand(u, p) = changeVarHyperCube(densityStepChange, u, p)

ExponentialGrowthIntegrand(u,p) = changeVarHyperCube(densityExponentialGrowth, u, p)



function fBranchLengthRatios(y, u::AbstractVector{Float64}, p)
    # p[1] = τ, p[2] = c
    ## The last two paramers of p are the indices i and j
    y[1] = changeVar(Integrand, u, p) 
end


function solveIntOOP(func, n, p, nbins=300, ncalls=5000)
    #OOP is Out of Place
    lower_terminals = zeros(n)
    upper_terminals = ones(n)
    domain = (lower_terminals, upper_terminals)

    integrand(u, p) = changeVarHyperCube(func, u, p)

    problem = IntegralProblem(integrand, domain, p)

    sol= solve(problem, VEGAS(; nbins=nbins, ncalls=ncalls))

    return sol.u
end

function solveIntIP_CubatureJLh(func, n, p, reltol)
    #the length of the param vector p will depend on the function func

    lower_terminals = zeros(n)
    upper_terminals = ones(1)
    domain = (lower_terminals, upper_terminals)

    prototype = zeros(1)

    problem = IntegralProblem(IntegralFunction(func, prototype), domain, p)

    sol= solve(problem, CubatureJLh(); reltol=reltol)

    return sol.u
end 

function solveIntIP_HyperCube(func, n, p, reltol)
    #the length of the param vector p will depend on the function func

    lower_terminals = zeros(n)
    upper_terminals = ones(n)
    domain = (lower_terminals, upper_terminals)

    # prototype = zeros(Int(n*(n-1)/2+n))
    

    problem = IntegralProblem(func, domain, p)

    sol= solve(problem, CubaVegas(); reltol=reltol)

    return sol.u
end 

function makeCorrelMatIP(method, n, τ, c, reltol)
    correlMat = zeros(n-1, n-1)

    if method == "VEGAS"
    for i = 1:n-1
        for j = 1:i
            correlMat[i,j] = solveIntIP_CubatureJLh(IntegrandSemiInfinite, n-1, [τ, c, i, j], reltol)[1]
            i != j ? correlMat[j,i] = correlMat[i,j] : nothing
        end
    end
    elseif method == "CubaVEGAS"
        vec = solveIntIP_HyperCube(VectorIntegrandHyperCube, n, [τ, c], reltol)
        correlMat = lowerTriangularMat(n, vec)
    else 
        println("Method not recognized")
    end 


    return correlMat
end



function makeProbMat(method, n, τ, c, reltol)
    correlMat = makeCorrelMatIP(method, n, τ, c, reltol)

    probMat = zeros(n-1, n-1)

    for i = 1:n-1
        for j = 1:i

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

function lowerTriangularMat(n, vec)
    mat = zeros(n,n)

    for i = 1:n
        k = i >= 2 ? sum(1:i-1) : 0
        for j = 1:i
            mat[i,j] = vec[k+j]
        end
    end 

    return mat
end

