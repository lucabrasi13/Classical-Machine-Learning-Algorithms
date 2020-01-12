%%%
% MATLAB test script to evaluate the performance of Bayes Classifier
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

%% Run the Bayes Classifier
K = 2;                      % Two class problem
alpha = 1;                  % Bump the diagonal to avoid singularity
[ytrain,ytest] = bayesClassifier(train,test,K,alpha);

%% Compute the accuracy on train and test
ltrain = ones(size(train,1),1);
ltrain(2:2:end) = 2;
ltest = ones(size(test,1),1);
ltest(2:2:end) = 2;
disp('The accuracy of the Bayes Classifier on training is = ');
disp(computeAccuracy(ltrain,ytrain));
disp('The accuracy of the Bayes Classifier on test is = ');
disp(computeAccuracy(ltest,ytest));
disp('The confusion Matrix of the test set is = ');
disp(confusionmat(ltest,ytest));
