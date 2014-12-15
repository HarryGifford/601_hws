function [cost, grad, delta] = costSVM(theta, X, y)
%COSTLR 1 vs all SVM cost function.
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
    
    y = 2*y - 1;
    xt = X*theta;
    margin = max(0, 1 - y.*xt);
    cost = 1./m*sum(margin(:));
    delta = -1./m*y.*(margin > 0);
    grad = X'*delta;
    
    %% END SOLUTION
    assert(numel(cost) == 1);
    assert(all(size(grad) == size(theta)));
    
    grad = grad(:);
end
