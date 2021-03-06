%%%
% MATLAB test script to evaluate the performance of Kernel SVM
%%%
%% Load the data
clc;clear;close;
dir = '/home/pheonix13/Documents/Books/UMCP/ENTS669E/Project/Project 1/Data/data.mat';
load(dir);

%% Data Pre-processing
X = reshape(face,[],600);
X = X';
X = zscore(X);
X(3:3:end,:) = [];

%% Dimensionality reduction
tol = 0.95;
U = PCA(X,tol);
Xeff = X*U;

%% Divide the dataset into train and test
train = Xeff(1:300,:);
test = Xeff(301:400,:);
M = size(train,1);
y = ones(M,1);
y(2:2:end) = -1;

%% Compute the Kernel Matrix
disp('Kernel SVM with RBF Kernel');
disp('Begin Training....');
v = 0.00045;
K(M,M) = 0;
for i = 1:M
    for j = 1:M
        K(i,j) = exp(-v*norm(train(i,:)-train(j,:))^2);
    end
end

%% Regularization parameter is found using line search
C = 3.9;

%% Compute the support vectors and the bias
[a,theta0] = kernelSVM(train,y,K,C,v);

%% Compute the train accuracy
decisionTrain(M,1) = 0;
for i = 1:M
    temp = 0;
    for j = 1:M
        temp = temp + a(j)*y(j)*K(i,j);  
    end
    if((temp+theta0)>1)
        decisionTrain(i) = +1;
    else
        decisionTrain(i) = -1;
    end
end

%% Compute the test accuracy
d2 = size(test,1);    
decisionTest(d2) = 0;
yT = ones(d2,1);
yT(2:2:end) = -1;
for i = 1:d2
    temp = 0;
    for j = 1:M
        temp = temp + a(j)*y(j)*exp(-v*norm(test(i,:)-train(j,:))^2);
    end
    temp = temp + theta0;
    if(temp >= 1)
        decisionTest(i) = +1;
    else
        decisionTest(i) = -1;
    end
end

%% Display the performance
disp('The accuracy of Kernel SVM with RBF kernel on training is = ');
disp(computeAccuracy(y,decisionTrain));

disp('The confusion Matrix is, ');
disp(confusionmat(y,decisionTrain));

disp('-------------------------------------------------------------');

disp('The accuracy of Kernel SVM with RBF kernel on test data is = ');
disp(computeAccuracy(yT,decisionTest));

disp('The confusion Matrix is, ');
disp(confusionmat(yT,decisionTest))

