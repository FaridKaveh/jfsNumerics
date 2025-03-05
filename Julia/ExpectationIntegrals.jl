using Integrals, LinearAlgebra 
using Cubature


PI = 3.14159265358979323846
setprecision(300)
f(u, p) = u[1]*u[2]*prod(p[1:2]) * exp(sum(-u .* p))/(sum(u))^2


g(u, p)=  u[1]^2 * p[1] * exp(sum(-u .* p))/(sum(u))^2

marginal_f(u, p) = u[1]*p[1]*exp(sum(-u .* p))/sum(u)


#the approx functions are the LLN approximation of the expectation integrals.
f_approx(u,p) = u[1]*u[2]*prod(p[1:2])*prod(exp.(-u .* p[1:3]))/(sum(u)+p[4])^2

g_approx(u,p) = u[1]^2 * p[1]* prod(exp.(-u .* p[1:2]))/(sum(u)+p[3])^2

function solvef(p, reltol)

    domain = (zeros(3), [Inf, Inf, Inf])
    problem = IntegralProblem(f, domain, p)
    
    sol = solve(problem, HCubatureJL(); reltol=reltol)
    
    return sol.u
end 

function solveg(p, reltol)
    domain = (zeros(2), [Inf, Inf])
    problem = IntegralProblem(g, domain, p)

    sol = solve(problem, HCubatureJL(); reltol=reltol)

    return sol.u
end 

function solvef_approx(p, reltol)

    domain = (zeros(3), [Inf, Inf, Inf])
    problem = IntegralProblem(f_approx, domain, p)

    sol = solve(problem, CubatureJLh(); reltol=reltol)

    return sol.u
end 

function solveg_approx(p, reltol)
    domain = (zeros(2), [Inf, Inf])
    problem = IntegralProblem(g_approx, domain, p)

    sol = solve(problem, CubatureJLh(); reltol=reltol)

    return sol.u
end 


get_diag(n,k,l) = 1/2 .* setdiff(collect(n:-1:2), [k,l]) .- 1/2
Λ(n,k,l) = diagm(0 => -get_diag(n,k,l), 1 => get_diag(n,k,l)[1:end-1]);


μ(n, m, k,l) = sum([2/(x-1) for x in setdiff(collect(m:n), [k,l])])

function J(n, k, l, reltol=1e-8)

    eig_sys = eigen(Λ(n,k,l))

    if k == l
        param_tuples = [[(k-1)/2, -eig_sys.values[i]] for i in 1:length(eig_sys.values)]
        diagIntegrals = solveg.(param_tuples, reltol)
    else
        param_triples = [[(k-1)/2, (l-1)/2, -eig_sys.values[i]] for i in 1:length(eig_sys.values)]
        diagIntegrals = solvef.(param_triples, reltol)
    end 

    # println(diagIntegrals)

    return (diagIntegrals, eig_sys)

end

function Japprox(n, m, k, l, reltol=1e-8)

    #n is the number of samples 
    #m is the level to truncate the integration
   
    eig_sys = eigen(Λ(m, k, l))
    
    if k==l
        param_triples = [[(k-1)/2, -eig_sys.values[i], μ(n, m, k,l)] for i in 1:length(eig_sys.values)]
        diagIntegrals = solveg_approx.(param_triples, reltol)
    else 
        param_sets = [[(k-1)/2, (l-1)/2, -eig_sys.values[i], μ(n, m, k,l)] for i in 1:length(eig_sys.values)]
        diagIntegrals = solvef_approx.(param_sets, reltol)
    end

    return (diagIntegrals, eig_sys)
end  

function α_approx(m, k, l)
    # m is not the sample size, its the level where integration is truncated
    if (k ≤ m && l ≤ m)
        alpha = zeros(m - 3 + (k==l))
    elseif (k ≤ m || l ≤ m)
        alpha = zeros(m-2)
    else 
        alpha = zeros(m-1)
    end

    alpha[1]=-1
    return alpha
end

function α(n, k, l)

    alpha = zeros(n-3+ (k==l))
    alpha[1] = -1
    return alpha
end 

function getProb(n, k, l, reltol)
    (diagIntegrals, eig_sys) = J(n,k,l, reltol)
    return α(n,k,l)' * eig_sys.vectors * diagm(0 => eig_sys.values .* diagIntegrals) * inv(eig_sys.vectors)* ones(length(α(n,k,l))) 
end



function getApproxProb(n, m, k, l, reltol)
    #TODO: rewrite this

    #n is the number of samples 
    #m is the level to truncate the integration
    if n ≤ 10
        return getProb(n, k, l, reltol)
    end
    (diagIntegrals, eig_sys) = Japprox(n, m, k,l, reltol)

    return α_approx(m, k, l)' * eig_sys.vectors * diagm(0 => eig_sys.values .* diagIntegrals) * inv(eig_sys.vectors)* ones(length(α_approx(m,k,l)))
end 

function makeProbMat(n, reltol, approx; truncation_level = 10)

    #probFunc is the function that computes the probabilities, it takes three arguments: n, k, l

    if !approx
        probMat = zeros(n-1, n-1)
        for i = 1:n-1
            for j = 1:i
                probMat[i,j] = getProb(n, i+1, j+1, reltol)
                i != j ? probMat[j,i] = probMat[i,j] : nothing
            end
        end

    else
        probMat = zeros(n-1,n-1)
        for i = 1:n-1
            for j = 1:i
                probMat[i,j] = getApproxProb(n, truncation_level, i+1, j+1, reltol)
                i != j ? probMat[j, i] = probMat[i, j] : nothing
            end 
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
    stirling = log(sqrt(2*PI*(n+1))) * (n+1)*log(n+1) -(n+1)
    return exp(stirling)*I.integral
end 

function ratioCorrelation(n, i, j)
    return calculateExpectation(x -> x[i-1]*x[j-1], n-1)
end 

