function [a,theta0,errInd,decision] = LSVM(train,y,C,D)
%% Define a few parameters
    M = size(train,1);
    
%% Generate a Linear Kernel
    K = train*train';
    
%% Compute the support vectors using quadprog
    H = (y*y').*K;
    f = -M*D';
    Aeq = y';
    beq = 0;
    lb = zeros(M,1);
    ub = (C/M)*ones(M,1);
    a = quadprog(H,f,[],[],Aeq,beq,lb,ub);
    a = a(:);
    
%% Compute the bias
    theta0 = 0;
    for i = 1:M
        temp = 0;
        for j = 1:M
            temp = temp + a(j)*y(j)*train(i,:)*train(j,:)';  
        end
        theta0 = theta0 + (y(i) - temp);
    end
    theta0 = theta0/M;

%% Calculate the number of misclassifications
    decision(M,1) = 0;
    for i = 1:M
        temp = 0;
        for j = 1:M
            temp = temp + a(j)*y(j)*train(i,:)*train(j,:)';  
        end
        if((temp+theta0)>1)
            decision(i) = +1;
        else
            decision(i) = -1;
        end
    end
    C = confusionmat(y,decision);
    errCount = sum(sum(tril(C,-1)+triu(C,1)));

%% Find the index of Misclassifications
    errInd(errCount) = 0;
    i = 1;
    j = 1;
    while(errCount > 0)
        if(decision(i) ~= y(i))
            errInd(j) = i;
            errCount = errCount-1;
            j = j+1;
            i = i+1;
        else
            i = i+1;
        end 
    end
end