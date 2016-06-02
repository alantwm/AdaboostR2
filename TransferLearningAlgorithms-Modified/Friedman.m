function f = Friedman(dataset,d)
    a = normrnd(1,0.1*d,1,4);
    b = normrnd(1,0.1*d,1,5);
    c = normrnd(0,0.05*d,1,5);
    
    r = size(dataset,1);
    f = zeros(r,1);
    
    for i = 1:r
        x=dataset(i,:);
        f(i) = a(1)*10*sin(pi*(b(1)*x(1)+c(1))*(b(2)*x(2)+c(2))) + a(2)*20*...
            (b(3)*x(3)+c(3)-0.5)^2 + a(3)*10*(b(4)*x(4)+c(4)) + a(4)*5*(b(5)*x(5)+c(5)) ...
            + 0.5;
    end
end