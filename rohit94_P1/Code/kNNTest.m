%%%
% MATLAB test script to evaluate the performance of kNN Classifier with LDA
%%%
%% Load the data
clc;clear;close;
dir = '/home/pheonix13/Documents/Books/UMCP/ENTS669E/Project/Project 1/Data/data.mat';
load(dir);

%% Data Pre-processing
X = reshape(face,[],600);
X = X';
%X = zscore(X);
X(3:3:end,:) = [];

%% Dimensionality reduction
tol = 0.95;
U = PCA(X,tol);
Xeff = X*U;

%% Divide the dataset into train and test
train = Xeff(101:400,:);
test = Xeff(1:100,:);

%% Design the labels on test
y = ones(size(test,1),1);
y(2:2:end) = 2;


%% Run the kNN classifier for different values of K
K = 2;
ytest = kNN(train,test,K);
disp('The accuracy of the kNN on test for K = 2, is = ');
disp(computeAccuracy(y,ytest));

disp('Confusion Matrix of test set, K = 2,');
disp(confusionmat(y,ytest));

disp('-------------------------------------------------');
K = 4;
ytest = kNN(train,test,K);
disp('The accuracy of the kNN on test for K = 4, is = ');
disp(computeAccuracy(y,ytest));

disp('Confusion Matrix of test set, K = 4,');
disp(confusionmat(y,ytest));

disp('-------------------------------------------------');
K = 6;
ytest = kNN(train,test,K);
disp('The accuracy of the kNN on test for K = 6, is = ');
disp(computeAccuracy(y,ytest));

disp('Confusion Matrix of test set, K = 6,');
disp(confusionmat(y,ytest));

disp('-------------------------------------------------');
K = 10;
ytest = kNN(train,test,K);
disp('The accuracy of the kNN on test for K = 8, is = ');
disp(computeAccuracy(y,ytest));

disp('Confusion Matrix of test set, K = 8,');
disp(confusionmat(y,ytest));

%% Improve test accuracy using Linear Discriminant Analysis
[thetaF,x0] = LDA(train,1);
trainNew(size(train,1),1) = 0;
testNew(size(test,1),1) = 0;
for i = 1:length(trainNew)
    trainNew(i) = (train(i,:)-x0')*thetaF;
end
for i = 1:length(testNew)
    testNew(i) = (test(i,:)-x0')*thetaF;
end

%% Rerun kNN on the new separated dataset
disp('================================================');
disp('Run kNN + LDA for K = 2,4,6,8');
K = 2;
yLDA = kNN(trainNew,testNew,K);
disp('The accuracy of the LDA on test for K = 2, is = ');
disp(computeAccuracy(y,yLDA));

disp('Confusion Matrix of test set, K = 2,');
disp(confusionmat(y,yLDA));

disp('-------------------------------------------------');
K = 4;
yLDA = kNN(trainNew,testNew,K);
disp('The accuracy of the LDA on test for K = 4, is = ');
disp(computeAccuracy(y,yLDA));

disp('Confusion Matrix of test set, K = 4,');
disp(confusionmat(y,yLDA));

disp('-------------------------------------------------');
K = 6;
yLDA = kNN(trainNew,testNew,K);
disp('The accuracy of the LDA on test for K = 6, is = ');
disp(computeAccuracy(y,yLDA));

disp('Confusion Matrix of test set, K = 6,');
disp(confusionmat(y,yLDA));

disp('-------------------------------------------------');
K = 8;
yLDA = kNN(trainNew,testNew,K);
disp('The accuracy of the LDA on test for K = 8, is = ');
disp(computeAccuracy(y,yLDA));

disp('Confusion Matrix of test set, K = 8,');
disp(confusionmat(y,yLDA));

disp('End of Simulation');
