function [a,theta0] = kernelSVM(train,y,K,C,v)
    %% Find the support vectors using quadprog
    d1 = size(train,1);
    H = (y*y').*K;
    f = -ones(d1,1);
    Aeq = y';
    beq = 0;
    lb = zeros(d1,1);
    ub = C*ones(d1,1);
    a = quadprog(H,f,[],[],Aeq,beq,lb,ub);
    a = a(:);

    %% Find the bias
    theta0 = 0;
    for i = 1:d1
        temp = 0;
        for j = 1:d1
            temp = temp + a(j)*y(j)*exp(-v*norm(train(i,:)-train(j,:))^2);  
        end
        theta0 = theta0 + (y(i) - temp);
    end
    theta0 = theta0/d1;
end
