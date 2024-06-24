using Integrals, LinearAlgebra 

f(u, p) = u[1]*u[2]*prod(-p) * prod(exp.(u .* p))/(sum(u))^2


g(u, p)=  u[1]^2 * prod(-p) * prod(exp.(u .* p))/(sum(u))^2
 

function solvef(p)

    domain = (zeros(3), [Inf, Inf, Inf])
    problem = IntegralProblem(f, domain, p)
    
    sol = solve(problem, HCubatureJL(); reltol=1e-8, abstol=1e-8)
    
    return sol.u
end 

function solveg(p)
    domain = (zeros(2), [Inf, Inf])
    problem = IntegralProblem(g, domain, p)

    sol = solve(problem, HCubatureJL(); reltol=1e-8, abstol=1e-8)

    return sol.u
end 

get_diag(n,k,l) = 1/2 .* setdiff(collect(n:-1:2), [k,l]) .- 1/2
Λ(n,k,l) = diagm(0 => -get_diag(n,k,l), 1 => get_diag(n,k,l)[1:end-1]);

function J(n, k, l)

    eig_sys = eigen(Λ(n,k,l))

    if k == l
        param_tuple = [[-(k-1)/2, eig_sys.values[i]] for i in 1:length(eig_sys.values)]
        diagIntegrals = solveg.(param_tuple)
    else
        param_triples = [[-(k-1)/2, -(l-1)/2, eig_sys.values[i]] for i in 1:length(eig_sys.values)]
        diagIntegrals = solvef.(param_triples)
    end 

    # println(diagIntegrals)
    matIntegral = eig_sys.vectors * diagm(0 => diagIntegrals) * inv(eig_sys.vectors)

    return matIntegral

end

function alpha(n, k, l)

    alpha = zeros(n-3+ (k==l))
    alpha[1] = -1
    return alpha
end 

function getProb(n, k, l)
    return alpha(n,k,l)' * Λ(n,k,l) * J(n,k,l) * ones(length(alpha(n,k,l)))
end