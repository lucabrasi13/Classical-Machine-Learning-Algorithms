function a = computeAccuracy(y,yhat)
% NAME
%   computeAccuracy - Compute the accuracy of a classifier
% FUNCTION
%   a = computeAccuracy(y,yhat)
% DESCRIPTION
%   Find the accuracy of a classifier by using MATLABs 'confusionmat'
% INPUTS
%   y       (vec)       (Mx1)  Vector of true labels
%   yhat    (vec)       (Mx1)  Vector of predicted labels
% OUTPUT
%   a       (scalar)           Computed accuracy    
% AUTHOR
%   Rohit Kashyap , November 2018
    y = y(:);yhat = yhat(:);
    M = length(y);
    a = sum(diag(confusionmat(y,yhat))/M);
end