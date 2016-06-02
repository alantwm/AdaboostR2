function f = betafunc(x,m,n,t,S,distribution,norm_err)
    target = (m/(n+m)) + (t/(S-1))*(1 - m/(n+m));
    distribution(m+1:m+n) = distribution(m+1:m+n).*x.^norm_err;
    distribution = distribution/sum(distribution);
    f = abs(target - sum(distribution(1:m)));
end