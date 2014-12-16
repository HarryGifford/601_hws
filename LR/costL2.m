function [cost, grad] = costL2(theta, X, y, lambda, funObj)
%COSTL2 Adds L2 regularization to the LR cost function.
%  theta  - n*K x 1 parameter vector
%  X      - m x n design matrix with m training examples.
%  y      - m x k matrix of indicator labels.
%           e.g. if the labels are 1, 3, 1, 2 then we would get the
%           following indicator matrix:
%           [1 0 1 0;
%            0 0 0 1;
%            0 1 0 0]
%  lambda - regularization strength.
%  funObj - cost function to add L2 regularization to.
%
%  cost  - 1 x 1 cost at theta.
%  grad  - n*k x 1 gradient at theta.
%
%  Example usage:
%   [cost, grad] = costL2(theta, X_train, full(sparse(1:m, y_train, 1)),
%                         0.1, @costSE);
%
%   load ../data/bullseye.mat
%   [m, n] = size(X_train);
%   K = max(y_train(:)) - min(y_train(:)) + 1;
%   [cost, grad] = costL2(ones(n*K, 1), X_train, full(sparse(1:m, y_train,
%   1)), 0.5, @costLR);
%
%     cost =
%       7.1919
%     grad =
%       1.1179
%       1.1861
%       0.8543
%       1.2624
%       0.5249
%       1.2956
%
    [m, n] = size(X);
    theta = reshape(theta, n, []);
    lambda = lambda*ones(size(theta));
    lambda(1, :) = 0;
    theta = theta(:);
    lambda = lambda(:);
    
    if nargout < 2
        cost = funObj(theta, X, y);
        cost = cost + 1/(2*m)*sum(lambda.*theta.^2);
    else
        [cost, grad] = funObj(theta, X, y);
        cost = cost + 1/(2*m)*sum(lambda.*theta.^2);
        grad = grad + 1/m*lambda.*theta;
    end
end
