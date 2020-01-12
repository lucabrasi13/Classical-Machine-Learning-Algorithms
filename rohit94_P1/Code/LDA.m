function [thetaF,x0] = LDA(X,alpha)
    M = size(X,2);
    K = 2;
    R(M,M,K) = 0;
    u(K,M) = 0;
    for i = 1:2
        R(:,:,i) = cov(X(i:2:end,:))+alpha*eye(M);
        u(i,:) = mean(X(i:2:end,:));
    end
    thetaF = pinv(R(:,:,1)+R(:,:,2))*(u(1,:)'-u(2,:)');
    x0 = 0.5.*(u(1,:)'+u(2,:)');
end






