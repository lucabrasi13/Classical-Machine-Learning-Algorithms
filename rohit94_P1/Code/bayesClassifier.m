% NAME
%   bayesClassifier - Function for Bayes Classification
% FUNCTION
%   [ytrain,ytest] = bayesClassifier(train,test,K,alpha)
% DESCRIPTION
%   Generate the decision vector for both test and training data
% INPUTS
%   train       (mat)       (M1xN)  Matrix of M1 training data points
%   test        (mat)       (M2xN)  Matrix of M2 test data points
%   K           (scalar)            Scalar indicating number of classes 
%   alpha       (scalar)            Scalar indicating the bump to avoid 0 det. 
% OUTPUT
%   ytrain      (vector)    (M1x1)  Vector indicating the decision on train    
%   ytest      (vector)    (M2x1)  Vector indicating the decision on test
% AUTHOR
%   Rohit Kashyap , November 2018
function [ytrain,ytest] = bayesClassifier(train,test,K,tol)
    N = size(train,1);
    M = size(train,2);
    R(M,M,K) = 0;
    u(K,M) = 0;
    for i = 1:K
        R(:,:,i) = cov(train(i:K:end,:))+tol*eye(M);
        u(i,:) = mean(train(i:K:end,:));
    end
    ytrain(N,1) = 0;
    prob(K) = 0;
    for i = 1:N
        for j = 1:K
            prob(j) = mvnpdf(train(i,:)',u(j,:)',R(:,:,j));
        end
        [~,ytrain(i)] = max(prob); 
    end
    P = size(test,1);
    ytest(P,1) = 0;
    for i = 1:P
        for j = 1:K
            prob(j) = mvnpdf(test(i,:)',u(j,:)',R(:,:,j)+tol*eye(M));
        end
        [~,ytest(i)] = max(prob); 
    end
end