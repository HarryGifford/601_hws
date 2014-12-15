function [cost, grad] = costL2(theta, X, y, lambda, funObj)
%COSTL2 Adds L2 regularization to the LR cost function.
%  theta - n*K x 1 parameter vector
%  n - number of features
%  lambda - regularization strength.
%  funObj - cost function to add L2 regularization to.
%
%  cost  - 1 x 1 cost at theta.
%  grad  - n*k x 1 gradient at theta.
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

