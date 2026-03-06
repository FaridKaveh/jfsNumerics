import numpy as np
import mpmath as mp
from Xiprob import *

# Set default precision (in decimal digits)
mp.mp.dps = 12
#lower this to make mp integrate faster, at the cost of a little precision
maxdegree = 4
# -----------------------
# ——— Core density functions
# -----------------------

def f(u, p):
    """
    f(u, p) = u1*u2*prod(p[0:2]) * exp(-u·p) / (sum(u))^2
    u: sequence of length 3
    p: sequence of length ≥3
    """
    u1, u2, u3 = u
    prod_p12 = p[0] * p[1]
    exponent = mp.e**(-(u1*p[0] + u2*p[1] + u3*p[2]))
    return u1 * u2 * prod_p12 * exponent / (u1 + u2 + u3)**2

def g(u, p):
    """
    g(u, p) = u1^2 * p1 * exp(-u·p) / (sum(u))^2
    u: sequence of length 2
    p: sequence of length ≥2
    """
    u1, u2 = u
    exponent = mp.e**(-(u1*p[0] + u2*p[1]))
    return u1**2 * p[0] * exponent / (u1 + u2)**2

def f_conditional(u, p):

    # vector of times
    u1, u2, u3 = u
    prod_p12 = p[0] * p[1]

    exponent = mp.e**(-(u1*p[0] + u2*p[1] + u3*p[2]))

    #mutation rate
    theta = p[3]
    mut_intensity = theta*(u1+u2+u3)/2
    #P(more than two mutations)
    xi_condition = 1-mp.exp(-mut_intensity)-mut_intensity*mp.exp(-mut_intensity)


    return u1 * u2 * prod_p12 * exponent * xi_condition / (u1 + u2 + u3)**2

def g_conditional(u, p):

    #time vector 
    u1, u2 = u

    #mutation rate
    theta = p[2]
    mut_intensity = theta * (u1+u2)/2
    #P(more than two mutations)
    xi_condition = 1-mp.exp(-mut_intensity)-mut_intensity*mp.exp(-mut_intensity)

    exponent = mp.e**(-(u1*p[0] + u2*p[1]))
   

    return u1**2 * p[0] * exponent*xi_condition/(u1 + u2)**2

def marginal_f(u, p):
    """
    marginal_f(u,p) = u1*p1*exp(-u·p)/sum(u)
    u: length 2
    p: length ≥2
    """
    u1, u2 = u
    exponent = mp.e**(-(u1*p[0] + u2*p[1]))
    return u1 * p[0] * exponent / (u1 + u2)

# ——— LLN‐approx versions
def f_approx(u, p):
    """
    f_approx(u,p) = u1*u2*prod(p[0:2])*prod(exp(-u*p[0:3]))/(sum(u)+p[3])^2
    """
    u1, u2, u3 = u
    prod_p12 = p[0] * p[1]
    exponent = mp.e**(-u1*p[0] - u2*p[1] - u3*p[2])
    return u1 * u2 * prod_p12 * exponent / (u1 + u2 + u3 + p[3])**2

def g_approx(u, p):
    """
    g_approx(u,p) = u1^2*p1*prod(exp(-u*p[0:2]))/(sum(u)+p[2])^2
    """
    u1, u2 = u
    exponent = mp.e**(-u1*p[0] - u2*p[1])
    return u1**2 * p[0] * exponent / (u1 + u2 + p[2])**2

# -----------------------
# ——— Numerical integrators
# -----------------------

def solvef(p, reltol=1e-8):
    """Integrate f over u1,u2,u3 in [0,∞)^3"""
    f_wrapped = lambda u1, u2, u3: f((u1, u2, u3), p)
    # nested integrals
    return mp.quad(
        lambda u1: mp.quad(
            lambda u2: mp.quad(
                lambda u3: f_wrapped(u1,u2,u3), 
                [0, mp.inf], 
                maxdegree=maxdegree),
            [0, mp.inf],
            maxdegree=maxdegree
        ),
        [0, mp.inf],
        error=reltol,
        maxdegree=maxdegree,
    )

def solvef_conditional(p, reltol=1e-8): 
    """Integrate f_modified over u1,u2,u3 in [0,∞)^3
       p: parameter list of length 4, fourth element is the mutation rate
    """
    f_wrapped = lambda u1, u2, u3: f_conditional((u1, u2, u3), p)

    return mp.quad(
        lambda u1: mp.quad(
            lambda u2: mp.quad(lambda u3: f_wrapped(u1,u2,u3), [0, mp.inf]),
            [0, mp.inf]
        ),
        [0, mp.inf],
        error=reltol,
    )

def solveg_conditional(p, reltol=1e-8): 
    """Integrate g_modified over u1,u2,u3 in [0,∞)^3
       p: parameter list of length 3, third element is the mutation rate
    """
    g_wrapped = lambda u1, u2: g_conditional((u1, u2), p)

    return mp.quad(
        lambda u1: mp.quad(lambda u2: g_wrapped(u1,u2), [0, mp.inf]),
        [0, mp.inf],
        error=reltol
    )

def solveg(p, reltol=1e-8):
    """Integrate g over u1,u2 in [0,∞)^2"""
    g_wrapped = lambda u1, u2: g((u1, u2), p)
    return mp.quad(
        lambda u1: mp.quad(lambda u2: g_wrapped(u1,u2), [0, mp.inf], maxdegree=maxdegree),
        [0, mp.inf],
        error=reltol,
        maxdegree=maxdegree
    )

def solvef_approx(p, reltol=1e-8):
    fa = lambda u1, u2, u3: f_approx((u1,u2,u3), p)
    return mp.quad(
        lambda u1: mp.quad(
            lambda u2: mp.quad(lambda u3: fa(u1,u2,u3), [0, mp.inf]),
            [0, mp.inf]
        ),
        [0, mp.inf],
        error=reltol
    )

import mpmath as mp

def solvef_approx_fast(p, reltol=1e-6):
    """
    Fast 3D integral of f_approx over [0,∞)^3 via
    u_i = t_i/(1-t_i), t_i∈[0,1], with explicit unpacking.
    """
    # Original integrand alias
    def fa(u1, u2, u3):
        return f_approx((u1, u2, u3), p)

    # Transformed integrand on [0,1]^3
    def integrand(t1, t2, t3):
        u1 = t1 / (1 - t1)
        u2 = t2 / (1 - t2)
        u3 = t3 / (1 - t3)
        # Jacobian = ∏ (1-t_i)^(-2)
        J = (1 - t1)**(-2) * (1 - t2)**(-2) * (1 - t3)**(-2)
        return fa(u1, u2, u3) * J

    # innermost: integrate over t3, unpack immediately
    def inner2(t1, t2):
        raw = mp.quad(
            lambda t3: integrand(t1, t2, t3),
            [0, 1],
            error=reltol
        )
        return raw[0] if isinstance(raw, tuple) else raw

    # middle: integrate inner2 over t2, unpack immediately
    def inner1(t1):
        raw = mp.quad(
            lambda t2: inner2(t1, t2),
            [0, 1],
            error=reltol
        )
        return raw[0] if isinstance(raw, tuple) else raw

    # outermost: integrate inner1 over t1, unpack immediately
    raw_outer = mp.quad(
        inner1,
        [0, 1],
        error=reltol
    )
    return raw_outer[0] if isinstance(raw_outer, tuple) else raw_outer


def solveg_approx(p, reltol=1e-8):
    ga = lambda u1, u2: g_approx((u1,u2), p)
    return mp.quad(
        lambda u1: mp.quad(lambda u2: ga(u1,u2), [0, mp.inf]),
        [0, mp.inf],
        error=reltol
    )


def solveg_approx_fast(p, reltol=1e-6):
    """
    Fast 2D integral of g_approx over [0,∞)^2 by t->u substitution,
    with explicit unpacking of both inner and outer mp.quad().
    """

    def ga(u1, u2):
        return g_approx((u1, u2), p)

    def integrand(t1, t2):
        u1 = t1 / (1 - t1)
        u2 = t2 / (1 - t2)
        J = (1 - t1)**(-2) * (1 - t2)**(-2)
        return ga(u1, u2) * J

    # first unpack the inner quad:
    def inner(t1):
        raw_inner = mp.quad(lambda t2: integrand(t1, t2),
                            [0, 1],
                            error=reltol)
        # unpack (value, err) → value
        return raw_inner[0] if isinstance(raw_inner, tuple) else raw_inner

    # now unpack the outer quad:
    raw_outer = mp.quad(inner,
                        [0, 1],
                        error=reltol)
    return raw_outer[0] if isinstance(raw_outer, tuple) else raw_outer


# -----------------------
# ——— Matrix and spectral functions
# -----------------------

def get_diag(n, k, l):
    """
    Julia: 1/2 .* setdiff(collect(n:-1:2), [k,l]) .- 1/2
    """
    seq = list(range(n, 1, -1))
    for x in (k, l):
        if x in seq:
            seq.remove(x)
    return [0.5*(x - 1) for x in seq]

def Lambda(n, k, l):
    """Construct the Λ matrix as in Julia code."""
    diag0 = [-d for d in get_diag(n, k, l)]
    diag1 = get_diag(n, k, l)[:-1]
    size = len(diag0)
    M = mp.matrix(size)
    # main diagonal
    for i in range(size):
        M[i,i] = diag0[i]
    # superdiagonal
    for i in range(size-1):
        M[i, i+1] = diag1[i]
    return M

def mu(n, m, k, l):
    """μ(n,m,k,l) = sum(2/(x-1) for x in setdiff(m:n,[k,l]))"""
    vals = [x for x in range(m, n+1) if x not in (k, l)]
    return mp.nsum(lambda i: 2/(i-1), [min(vals), max(vals)])  # or direct sum

# -----------------------
# ——— J and Japprox
# -----------------------

def J(n, k, l, reltol=1e-8):
    Λ = Lambda(n, k, l)
    eigvals, eigvecs = mp.eig(Λ)  # returns (values, vectors)
    diagIntegrals = []
    if k == l:
        for eig in eigvals:
            params = [(k-1)/2, -eig]
            diagIntegrals.append(solveg(params, reltol))
    else:
        for eig in eigvals:
            params = [(k-1)/2, (l-1)/2, -eig]
            diagIntegrals.append(solvef(params, reltol))
    return diagIntegrals, (eigvals, eigvecs)

def J_conditional(n, k, l, theta, reltol=1e-8):
    Λ = Lambda(n, k, l)
    eigvals, eigvecs = mp.eig(Λ)  # returns (values, vectors)
    diagIntegrals = []
    if k == l:
        for eig in eigvals:
            params = [(k-1)/2, -eig, theta]
            diagIntegrals.append(solveg_conditional(params, reltol))
    else:
        for eig in eigvals:
            params = [(k-1)/2, (l-1)/2, -eig, theta]
            diagIntegrals.append(solvef_conditional(params, reltol))
    return diagIntegrals, (eigvals, eigvecs)

def Japprox(n, m, k, l, reltol=1e-8):
    Λ = Lambda(m, k, l)
    eigvals, eigvecs = mp.eig(Λ)
    diagIntegrals = []
    if k == l:
        for eig in eigvals:
            params = [(k-1)/2, -eig, mu(n, m, k, l)]
            diagIntegrals.append(solveg_approx_fast(params, reltol))
    else:
        for eig in eigvals:
            params = [(k-1)/2, (l-1)/2, -eig, mu(n, m, k, l)]
            diagIntegrals.append(solvef_approx_fast(params, reltol))
    return diagIntegrals, (eigvals, eigvecs)

# -----------------------
# ——— α and α_approx
# -----------------------

def alpha(n, k, l):
    size = n - 3 + (1 if k == l else 0)
    arr = mp.matrix([0]*size)
    arr[0] = -1
    return arr

def alpha_approx(m, k, l):
    if k <= m and l <= m:
        size = m - 3 + (1 if k == l else 0)
    elif k <= m or l <= m:
        size = m - 2
    else:
        size = m - 1
    arr = mp.matrix([0]*size)
    arr[0] = -1
    return arr

# -----------------------
# ——— Probability computations
# -----------------------

def getProb(n, k, l, reltol=1e-8):
    diagInts, (eigvals, eigvecs) = J(n, k, l, reltol)
    A = alpha(n, k, l).T
    
    D = mp.diag([eigvals[i] * diagInts[i][0] for i in range(len(eigvals))])
    V = eigvecs
    invV = V**-1
    ones = mp.matrix([1]*len(eigvals))
    return A * V * D * invV * ones

def getConditionalExpectation(n, k, l, theta, reltol=1e-8): 

    #note diagInts is a list of tuples of the form (mp.float, mp.float)
    #the first elemnt is the integral value, the second is the (absolute?) error.

    diagInts, (eigvals, eigvecs) = J_conditional(n, k, l, theta, reltol)
    A = alpha(n, k, l).T
    print(diagInts)
    D = mp.diag([eigvals[i] * diagInts[i][0] for i in range(len(eigvals))])
    V = eigvecs
    invV = V**-1
    ones = mp.matrix([1]*len(eigvals))
    return A * V * D * invV * ones

def getApproxProb(n, m, k, l, reltol=1e-8):
    if n <= 10:
        return getProb(n, k, l, reltol)
    diagInts, (eigvals, eigvecs) = Japprox(n, m, k, l, reltol)

    clean_ints = []
    for di in diagInts:
        if isinstance(di, (list, tuple)):
            clean_ints.append(mp.mpf(di[0]))
        else:
            clean_ints.append(mp.mpf(di))

    A = alpha_approx(m, k, l).T
    print(type(eigvals[0]), type(clean_ints[0]), diagInts[0])
    D = mp.diag([eigvals[i] * clean_ints[i] for i in range(len(eigvals))])
    V = eigvecs
    invV = V**-1
    ones = mp.matrix([1]*len(eigvals))
    return A * V * D * invV * ones

def makeProbMat(n, reltol=1e-8, mode="conditional", theta=5, truncation_level=10):
    """
    Constructs an (n-1)x(n-1) symmetric probability matrix.
    If approx=True, uses getApproxProb; otherwise getProb.
    """
    mat = mp.matrix(n-1)

    for i in range(n-1):
        for j in range(i+1):

            if mode == "approximate":
                val = getApproxProb(n, truncation_level, i+2, j+2, reltol)

            elif mode == "unconditional":
                val = getProb(n, i+2, j+2, reltol)

            elif mode == "conditional": 
                val = getConditionalExpectation(n, i+2, j+2, theta, reltol)

            else:
                raise ValueError(f"undefined mode: {mode}")
            
            val = val[0,0]
            mat[i,j] = val
            mat[j,i] = val

    if mode == "conditional": 
        mat = mat*(1/prob_m_g_two(theta, n))

    return mat
