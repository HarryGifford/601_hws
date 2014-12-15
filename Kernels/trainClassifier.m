function opt = trainClassifier(X, y, opt)
% train classifier
%
%  X      - m x n design matrix
%  y      - m x 1 labels
%  opt    - options
%     .loss     - 'mlr' for softmax regression and 'l2svm' for L2 SVM.
%                 Default is 'mlr'.
%
%     .lambda   - regularization parameter. Default is 0.
%
%     .dual     - optimize in the dual if true. Default is false. If false
%                 then a linear kernel is used.
%
%     .kernelfn - kernel function - Either a string 'rbf' for RBF kernel or
%                 'poly' for a polynomial kernel.
%                 Alternatively, kernelfn can be a function kernelfn(x, y)
%                 which should return an m1 x m2 gram matrix between 
%                 x and y, where there are m1 examples in x and m2 in y.
%                 For example you can implement a tanh kernel with params
%                 a and b as opt.kernelfn = @(X1, X2) tanh(a*X1*X2' - b).
%
%     .gamma    - RBF kernel width. Larger gamma => smaller variance.
%                 gaussian. Default = 1.
%
%     .order    - Polynomial order. Default = 3.
%
%  opt - updated options containing theta
%     .theta - (m+1)*K x 1 parameter vector
%

    if nargin < 3, opt.dual = false; end
    opt = handleOptions(opt);
    
    m = size(X, 1);
    K = max(y);
    
    if opt.dual
        opt.X = X;
        X = opt.kernelfn(X, X);
    else
        X = [ones(m, 1), X];
    end
    
    theta = zeros(size(X, 2), K); % initialize the parameters to zero.

    y = full(sparse(1:m, y, 1)); % create an indicator matrix of labels.
        
    opt.theta = minFunc(@l2regu, theta(:), opt, X, y, opt);
    opt.theta = reshape(opt.theta, size(X, 2), []);
end

function opt = handleOptions(opt)
    if ~isfield(opt, 'dual'), opt.dual = false; end
    if ~isfield(opt, 'loss'), opt.loss = 'mlr'; end
    if strcmp(opt.loss, 'mlr'), opt.loss = @costMLR;
    else opt.loss = @costL2SVM; end
    if ~isfield(opt, 'lambda'), opt.lambda = 0; end
    if ~isfield(opt, 'display'), opt.display = true; end
    if opt.dual
        if ~isfield(opt, 'kernelfn') || strcmp(opt.kernelfn, 'rbf')
            if ~isfield(opt, 'gamma'), opt.gamma = 1; end
            opt.kernelfn = @(x, y) rbfKernel(x, y, opt.gamma);
        elseif strcmp(opt.kernelfn, 'poly')
            if ~isfield(opt, 'order'), opt.order = 2; end
            opt.kernelfn = @(x, y) polyKernel(x, y, opt.order);
        end
    end
end

function [cost, grad, delta] = l2regu(theta, X, y, opt)
% Adds L2 regularization to opt.loss's cost function.
%  theta - parameter vector
%  X     - m x n design matrix
%  y     - m x K labels
%  opt   - options
%     .loss   - cost function to add regularization to
%     .lambda - regularization strength
%     .dual   - true for dual problem, false for primal
%
%  cost  - 1 x 1 cost at theta
%  grad  - m x 1 gradient at theta
%  delta - error used in backpropagation
%
    [m, n] = size(X);
    theta = reshape(theta, n, []);

    [cost, grad, delta] = opt.loss(theta, X, y);
    if opt.dual
        cost = cost + opt.lambda./(2*m).*sum(sum(theta'*X*theta));
        kt = X*theta;
        grad = grad + opt.lambda./m.*kt(:);
    else
        cost = cost + opt.lambda./(2*m).*sum(sum(theta(2:end, :).^2));
        lambda = opt.lambda.*ones(size(theta));
        lambda(1, :) = 0;
        grad = grad + lambda(:).*theta(:)./m;
    end
end

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


