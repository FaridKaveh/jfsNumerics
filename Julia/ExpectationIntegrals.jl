using Integrals, LinearAlgebra 

f(u, p) = u[1]*u[2]*prod(p[1:2]) * exp(sum(-u .* p))/(sum(u))^2


g(u, p)=  u[1]^2 * p[1] * exp(sum(-u .* p))/(sum(u))^2


#the approx functions are the LLN approximation of the expectation integrals.
f_approx(u,p) = u[1]*u[2]*prod(exp.(-u .* p[1:2]))/(sum(u)+p[3])^2

g_approx(u,p) = u[1]^2 * prod(exp.(-u .* p[1]))/(sum(u)+p[2])^2

function solvef(p)

    domain = (zeros(3), [Inf, Inf, Inf])
    problem = IntegralProblem(f, domain, p)
    
    sol = solve(problem, HCubatureJL(); reltol=1e-6, abstol=1e-6)
    
    return sol.u
end 

function solveg(p)
    domain = (zeros(2), [Inf, Inf])
    problem = IntegralProblem(g, domain, p)

    sol = solve(problem, HCubatureJL(); reltol=1e-6, abstol=1e-6)

    return sol.u
end 

function solvef_approx(p)

    domain = (zeros(2), [Inf, Inf])
    problem = IntegralProblem(f_approx, domain, p)

    sol = solve(problem, HCubatureJL(); reltol=1e-10, abstol=1e-10)

    return sol.u
end 

function solveg_approx(p)
    domain = (zeros(1), [Inf])
    problem = IntegralProblem(g_approx, domain, p)

    sol = solve(problem, HCubatureJL(); reltol=1e-10, abstol=1e-10)

    return sol.u
end 


get_diag(n,k,l) = 1/2 .* setdiff(collect(n:-1:2), [k,l]) .- 1/2
Λ(n,k,l) = diagm(0 => -get_diag(n,k,l), 1 => get_diag(n,k,l)[1:end-1]);


μ(n,k,l) = sum([1/x for x in get_diag(n,k,l)])

function J(n, k, l)

    eig_sys = eigen(Λ(n,k,l))

    if k == l
        param_tuples = [[(k-1)/2, -eig_sys.values[i]] for i in 1:length(eig_sys.values)]
        diagIntegrals = solveg.(param_tuples)
    else
        param_triples = [[(k-1)/2, (l-1)/2, -eig_sys.values[i]] for i in 1:length(eig_sys.values)]
        diagIntegrals = solvef.(param_triples)
    end 

    # println(diagIntegrals)

    return (diagIntegrals, eig_sys)

end

function α(n, k, l)

    alpha = zeros(n-3+ (k==l))
    alpha[1] = -1
    return alpha
end 

function getProb(n, k, l)
    (diagIntegrals, eig_sys) = J(n,k,l)
    return α(n,k,l)' * eig_sys.vectors * diagm(0 => eig_sys.values .* diagIntegrals) * inv(eig_sys.vectors)* ones(length(α(n,k,l))) 
end

function getApproxProb(n, k, l)
    if k == l
        return solveg_approx([(k-1)/2, μ(n,k,l)])
    else
        return solvef_approx([(k-1)/2, (l-1)/2, μ(n,k,l)])
    end
end 

function makeProbMat(n, probFunc)

    #probFunc is the function that computes the probabilities, it takes three arguments: n, k, l
    probMat = zeros(n-1, n-1)
    for i in 1:n-1
        for j in 1:i
            probMat[i,j] = probFunc(n, i+1, j+1)
            i != j ? probMat[j,i] = probMat[i,j] : nothing
        end
    end

    return probMat
end



function approxProbMat(n) 

    probMat = zeros(n-1, n-1)

    for i in 1:n-1
        for j in 1:i
            probMat[i,j] = getApproxProb(n, i+1, j+1)
            i != j ? probMat[j,i] = probMat[i,j] : nothing
        end
    end

    return probMat
end 


#Now I'm gonna use the alternative form of the expectation integrals to compute the probabilities and compare performance.

using SimplicialCubature #needed to integrate on the simplex

S(n) = CanonicalSimplex(n)
get_rates(n) = 1/2 .* collect(n:-1:2) .- 1/2

function ratioDensity(x)
    n = length(x)+1
    rates = get_rates(n+1) 
    diffs = rates[1:n-1] .- rates[n].*ones(n-1)
    return prod(rates)/(rates[n]+dot(reverse(x), diffs))^n
end

function calculateExpectation(g, n)

    function f(x)
        return g(x) * ratioDensity(x)
    end

    I = integrateOnSimplex(f, S(n), maxEvals=1e8, tol=1e-6)
    return factorial(n+1)*I.integral
end 

function ratioCorrelation(n, i, j)
    return calculateExpectation(x -> x[i-1]*x[j-1], n-1)
end 