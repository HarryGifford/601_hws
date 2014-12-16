function opt = trainLR(X, y, opt)
%TRAINLR Trains logistic regression classifier
%  X - m x n design matrix.
%  y - m x 1 vector of labels.
%  opt - options
%     .lambda  - regularization strength
%     .MaxIter - maximum number of iterations for minimization.
%
    addpath(genpath('../shared/'));
    [m, n] = size(X);
    opt.ymin = min(y);
    y = y - opt.ymin + 1;
    K = max(y);
    y = full(sparse(1:m, y, 1));
    
    % check the gradient on a toy example
    toylambda = 4.0;
    toyn = 3;
    toym = 10;
    toyK = 2;
    toyw = zeros(toyn, toyK);
    toyX = rand(toym, toyn);
    toyY = rand(toym, 1) < 0.5;
    toyY = [toyY, ~toyY];
    % first check gradient of unregularized cost function.
    checkGradient(@(t) costLR(t(:), toyX, toyY), toyw(:));
    % now check gradient of regularized cost function.    
    checkGradient(@(t) costL2(t(:), toyX, toyY, toylambda, @costLR), toyw(:));
    
    X = [ones(m, 1), X];
    
    theta = zeros(n+1, K);
    theta = minimize(@(t, X, y) costL2(t, X, y, opt.lambda, @costLR),...
                     theta(:), X, y, opt);
    opt.theta = reshape(theta, n+1, K);

end

