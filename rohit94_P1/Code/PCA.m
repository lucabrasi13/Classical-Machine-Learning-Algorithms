function U = PCA(X,tol)
%% Center the data
    %X = X-repmat(mean(X),size(X,1),1);
%% Spectral decomposition to find the principal components
    [U,V] = eig(cov(X));
    V = diag(V);
%% Sort the Eigenvectors according to decreasing eigenvalues    
    [~,ind] = sort(-1*V);
    V = V(ind);
    U = U(:,ind);

%% Select the eigen values based on tolerence
    M = 0;
    flag = Inf;
    i = 1;
    while(flag~=0)
        if(sum(V(1:i))/sum(V)>=tol)
            flag = 0;
        else
            M = M+1;
            i = i+1;
        end
    end
%% Return the M principal components
    U = U(:,1:M);
end