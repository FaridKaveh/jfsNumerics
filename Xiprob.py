def prob_m_g_two(theta: float, n: int)-> float:
    ls1 = [(i-1)/(i-1+theta) for i in range(2,n+1)]
    ls2 = [theta/(i-1+theta) for i in range(2,n+1)]

    fac1 = 1
    for term in ls1:
        fac1 *= term
    fac2 = 1 + sum(ls2)

    return 1-fac1*fac2

