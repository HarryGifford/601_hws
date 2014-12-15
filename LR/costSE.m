function [cost, grad, delta] = costSE(theta, X, y)
%COSTSE 1 vs all linear regression cost function.
%  theta - n*k x 1 parameter vector.
%  X     - m x n design matrix.
%  y     - m x k indicator matrix of labels.
%           e.g. if the labels are 1, 3, 1, 2 then we would get the
%           following indicator matrix:
%           [1 0 1 0;
%            0 0 0 1;
%            0 1 0 0]
% 
%  cost  - 1 x 1 cost at theta.
%  grad  - n*k x 1 gradient at theta.
%  delta - m*k x 1 error used for backpropagation.
%
    [m, n] = size(X);
    theta = reshape(theta, n, []);
    
    cost = 0;
    grad = zeros(size(theta));
    delta = zeros(m, size(theta, 2));
    
    %% BEGIN SOLUTION
    
    xt = X*theta;
    diff = xt - y;
    cost = 1/(2*m)*sum(diff(:).^2);
    delta = 1/m*diff;
    grad = X'*delta;
    
    %% END SOLUTION
    assert(numel(cost) == 1);
    assert(all(size(grad) == size(theta)));
    
    grad = grad(:);
end

function y = sigmoid(x)
    y = 1./(1+exp(-x));
end