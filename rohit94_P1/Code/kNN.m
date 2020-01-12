function ytest = kNN(train,test,K)
    M1 = size(train,1);
    M2 = size(test,1);
    Dist(M2,M1) = 0;
    kClosest(M2,K) = 0;
    ytest(M2) = 0;
    for i = 1:M2
        for j = 1:M1
            Dist(i,j) = norm(train(j,:) - test(i,:));
        end
        [~,kClosest(i,:)] = mink(Dist(i,:),K);
        count = 0;
        for l = 1:K
            if(mod(kClosest(i,l),2)==1)
                count = count +1;
            end
        end
        if(count > (K-count))
            ytest(i) = 1;
        else
            ytest(i) = 2;
        end
    end
end