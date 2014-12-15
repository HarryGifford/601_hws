function [cost, grad, delta] = costL2SVM(theta, X, y)
% L2 SVM cost function. Similar to normal SVM, except we penalize the
% square of the slack variables. Cost for K 1 vs all SVMS.
%
%  theta  - m x 1 parameter vector
%  X      - m x n design matrix
%  y      - m x K indicator matrix of labels
%
%  cost  - 1 x 1 cost at theta
%  grad  - m x 1 gradient at theta
%  delta - error used in backpropagation
%
    [m, n] = size(X);
    theta = reshape(theta, n, []);
    y = 2*y - 1;
    
    % compute cost
    margin = max(0, 1 - y.*(X*theta));
    cost = 1/(2*m)*sum(margin(:).^2);
    
    if nargout < 2, return; end
    
    % compute grad
    delta = -1/m*margin.*y;
    grad = X'*delta;
    grad = grad(:);
end
