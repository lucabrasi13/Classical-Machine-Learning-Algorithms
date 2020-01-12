%%% 
% Matlab code to boost weak Linear SVM using Adaboost
%%%
clc;clear;close

%% Load the dataset
dir = '/home/pheonix13/Documents/Books/UMCP/ENTS669E/Project/Project 1/Data/data.mat';
load(dir);
clear dir

%% Dataset Pre-processing
X = reshape(face,[],600);
X = X';
X = zscore(X);
X(1:3:end,:) = [];

%% Dimensionality reduction
tol = 0.959;
U = PCA(X,tol);
Xeff = X*U;
train = Xeff(1:300,:);
test = Xeff(301:400,:);

%% Define a few parameters
C = 1.5;
N = size(train,1);
y = ones(N,1);
y(2:2:end) = -1;

%% Initialize the weights and define a few parameters for boosting
f1 = size(train,1);
numIter = 5;
a(numIter) = 0;
P(numIter) = 0;
w(numIter+1,N) = 0;             
w(1,:) = ones(f1,1)/f1;
Z(numIter) = 0;
D = ones(N,1);
sv(numIter,N) = 0;
theta0(numIter) = 0;

%% Run boosting
for i = 1:numIter
    [sv(i,:),theta0(i),Ix,decision] = LSVM(train,y,C,D);
    for k = 1:size(Ix)
        P(i) = P(i)+w(i,Ix(k));
    end
    a(i) = 0.5*log((1-P(i))/P(i));
    Z(i) = 0;
    for j = 1:N
        w(i+1,j) = w(i,j)*exp(-y(j)*a(i)*decision(j));
        Z(i) = Z(i) + w(i+1,j);
    end
    for l = 1:N
        w(i+1,l) = w(i+1,l)/Z(i);
    end
    D = w(i+1,:);
end

%% Evaluate the error on the training set
decisionTrain(size(train,1)) = 0;
for k = 1:size(train,1)
    temp2 = 0;
    for i = 1:numIter
        temp1 = 0;
        for j = 1:N
            temp1 = temp1+sv(i,j)*y(j)*train(k,:)*train(j,:)';
        end
        temp1 = temp1+theta0(i);
        temp2 = temp2+(a(i)/sum(a))*sign(temp1);
    end
    if(sign(temp2)==1)
        decisionTrain(k) = 1;
    else
        decisionTrain(k) = -1;
    end
end

%% Evaluate the error on the test set
temp1 = 0;
temp2 = 0;
decisionTest(size(test,1)) = 0;
yT = ones(size(test,1),1);
yT(2:2:end) = -1;
for k = 1:size(test,1)
    temp2 = 0;
    for i = 1:numIter
        temp1 = 0;
        for j = 1:N
            temp1 = temp1+sv(i,j)*y(j)*test(k,:)*train(j,:)';
        end
        temp1 = temp1+theta0(i);
        temp2 = temp2+(a(i)/sum(a))*sign(temp1);
    end
    if(sign(temp2)==1)
        decisionTest(k) = 1;
    else
        decisionTest(k) = -1;
    end
end

%% Display the result
C1 = confusionmat(y,decisionTrain);
disp('The accuracy on train set is,');
disp(sum(diag(C1))/300);

disp('The confusion Matrix is, ');
disp(C1);

disp('------------------------------------------');

C = confusionmat(yT,decisionTest);
disp('The accuracy on test set is,');
disp(sum(diag(C))/100);

disp('The confusion Matrix is, ');
disp(C);
