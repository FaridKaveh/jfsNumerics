# don't need this but adding for future compatibity, in case binomial moves from base
using Combinatorics 
using Integrals
using Cubature
using Cuba
using MonteCarloIntegration
using ArgCheck



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

    #we need high precision arithmetic because the summands can be very large
    setprecision(400)
    pushfirst!(s, 0)
    n = length(s)

    binomialCoeff = [binomial(j, 2) for j = 2:n] 
    logSummands = log.(binomialCoeff) .+ β*s[2:end]
    
    expTimes = exp.(BigFloat(β) * s)
    intSummands = expTimes[2:end] - expTimes[1:end-1]
    intSummands = intSummands .* binomialCoeff

    exponent = sum(intSummands .- logSummands)

    

    # return exponent ≤ 1e16 ? exp(-exponent) : 0.0
    return exp(-exponent)
    
end 


function densityStepChange(s, p::NTuple{2, Real})
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
    
    if any(isinf.(s))
        println(s)
    end 


    for j = 2:n
        intSummands[j-1] = (s[j]-s[j-1])*((j < stepPoint) + c*(j-1>=stepPoint)) + (τ-s[j-1] + c*(s[j]-τ))*(j == stepPoint)
        intSummands[j-1] *= binomial(n-j+2,2)
    end

    
    return exp(sum(logSummands .- intSummands))
end

function densityBottleneck(s, p::NTuple{4, Real})
    #this is the density function for a bottleneck population histroy
    #τ1 and τ2 give the times of the stepwise changes and c1 and c2 are the levels to which the 
    #population jumps 
    τ1 = p[1]; τ2 = p[2]
    c1= p[3]; c2 = p[4]

    @argcheck τ1 < τ2
    
    
    @argcheck !any(isnan.(s))

    pushfirst!(s,0)
    n = length(s)

    stepPoint1 = binSearch(τ1, s)
    stepPoint2 = binSearch(τ2, s)

    logSummands = [binomial(n-j+2, 2)*(1 - (1-c1)*(j≥stepPoint1 && j < stepPoint2) 
            - (1-c2)*(j ≥stepPoint2)) for j = 2:n]
    logSummands = log.(logSummands)

    intSummands = zeros(n-1)
 for j = 2:n

    debug_bool = [(j < stepPoint1), ((j-1 ≥ stepPoint1 && j < stepPoint2)), (j-1 ≥ stepPoint2), (j-1 < stepPoint1 && j ≥ stepPoint2), (j-1 ≥ stepPoint1 && j-1 < stepPoint2 && j ≥ stepPoint2),
     (j-1 < stepPoint1 && j ≥ stepPoint1 && j < stepPoint2)]

    # I don't know why but this code does not work if I write it as one single line
    if debug_bool[1]
        intSummands[j-1] = (s[j] - s[j-1])
    elseif debug_bool[2]
        intSummands[j-1] = c1*(s[j]-s[j-1])
    elseif debug_bool[3]
        intSummands[j-1] = c2*(s[j]-s[j-1])
    elseif debug_bool[4]
        intSummands[j-1] = ((τ1 - s[j-1])+c1*(τ2-τ1)+c2*(s[j]-τ2))
    elseif debug_bool[5]
        intSummands[j-1] = (c1*(τ2-s[j-1])+ c2*(s[j]- τ2))
    elseif debug_bool[6]
        intSummands[j-1] = (τ1 - s[j-1])+c1*(s[j]-τ1)

    end 
    # intSummands[j-1] = (s[j] - s[j-1])*(j < stepPoint1) 
    #     + c1*(s[j]-s[j-1])*(j-1 ≥ stepPoint1 && j < stepPoint2)
    #     + c2*(s[j]-s[j-1])*(j-1 ≥ stepPoint2) + ((τ1 - s[j-1])+c1*(τ2-τ1)+c2*(s[j]-τ2))*(j-1 < stepPoint1 && j ≥ stepPoint2)
    #     + (c1*(τ2-s[j-1])+ c2*(s[j]- τ2))*(j-1 ≥ stepPoint1 && j-1 < stepPoint2 && j ≥ stepPoint2)+ ((τ1 - s[j-1])+c1*(s[j]-τ1))*(j-1 < stepPoint1 && j ≥ stepPoint1 && j < stepPoint2)
        
    intSummands[j-1] *= binomial(n-j+2, 2)

end

    return exp(sum(logSummands .- intSummands))
end 

        

function branchLengthRatios(s::Vector{Float64}, i, j) 
    #This is the ratio S_i*S_j/(T_{tot}^\ast)^2
    i, j = Int(i), Int(j)
    n= length(s)
    s[n-i+1]*s[n-j+1]/(sum(s)+s[n])^2
end 

IntegrandStepChange(u, p) = densityStepChange(u, p[1:2])*branchLengthRatios(u, p[3], p[4])
IntegrandBottleNeck(u,p) = densityBottleneck(u, p[1:4]) * branchLengthRatios(u, p[5], p[6])

function VectorIntegrand(u, p::NTuple)
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
    tn = s[end] != 1 ? s[end]/(1-s[end]) : (1-1e-12)
    times = [tn*prod(s[j] for j in i:n-1) for i in 1:n-1]
    push!(times, tn)

    jacobian = prod([s[j]^(j-1) for j in 1:n-1]) * tn^(n-1) * 1/(1-s[end])^2
    
    if any(isinf.(times))
        println("s = ", s)
    end

    if s[end] == 1
        println("s[end] == 1")
    end

    return func(times, p) *jacobian
end

# IntegrandSemiInfinite(u, p) = changeVar(Integrand, u, p)

# IntegrandHyperCube(u, p) = changeVarHyperCube(Integrand, u, p)

# VectorIntegrandHyperCube(u,p) = changeVarHyperCube(VectorIntegrand, u, p)

# StepChangeIntegrand(u, p) = changeVarHyperCube(densityStepChange, u, p)

# ExponentialGrowthIntegrand(u,p) = changeVarHyperCube(densityExponentialGrowth, u, p)

# BottleneckIntegrand(u, p) = changeVarHyperCube(densityBottleneck, u, p)

BottleneckRatios(u, p) = densityBottleneck(u, p[1:4]) * branchLengthRatios(u, p[5:end]...)

function fBranchLengthRatios(y, u::AbstractVector{AbstractFloat}, p)
    # p[1] = τ, p[2] = c
    ## The last two paramers of p are the indices i and j
    y[1] = changeVar(Integrand, u, p) 
end


function solveIntVEGAS(func, n, p; nbins=300, ncalls=5000)
    
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
    upper_terminals = ones(n)
    domain = (lower_terminals, upper_terminals)


    integrand(u, p) = changeVarHyperCube(func, u, p)

    problem = IntegralProblem(integrand, domain, p)

    sol= solve(problem, CubatureJLh(); reltol=reltol)

    return sol.u
end 


function makeCorrelMatIP(density, n, params; nbins=300, ncalls=5000)

    integrand(u, p) = branchLengthRatios(u, p[1], p[2])*density(u, p[3:end]) 

    correlMat = zeros(n-1, n-1)

    for i = 1:n-1
        for j = 1:i
            correlMat[i,j] = solveIntVEGAS(integrand, n-1, (i,j,params...); nbins=nbins, ncalls=ncalls)
            i != j ? correlMat[j,i] = correlMat[i,j] : nothing
        end
    end 

    return correlMat
end





function makeProbMat(density, n, params; nbins=300, ncalls=5000)
    correlMat = makeCorrelMatIP(density, n, params; nbins, ncalls)

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

# solveIntIP_CubatureJLh(BottleneckIntegrand, 3, (1/2,1,2,1), 1e-6)