function [cost, grad, delta] = costMLR(theta, X, y)
% Multinomial logistic regression cost function.
%
%  theta  - n*K x 1 parameter vector
%  X      - m x n design matrix
%  y      - m x K indicator matrix of labels
%
%  cost  - 1 x 1 cost at theta
%  grad  - m x 1 gradient at theta
%  delta - error used in backpropagation
%
    [m, n] = size(X);
    
    % theta is overparameterized, but it's ok.
    theta = reshape(theta, n, []);
    
    % compute cost
    tx = X*theta;
    logpy = bsxfun(@minus, tx, logsumexp(tx,2));
    cost = -1/m*y(:)'*logpy(:);
    
    if nargout < 2, return; end
    
    % compute grad
    delta = 1/m*(exp(logpy) - y);
    grad = X'*delta;
    grad = grad(:);
end