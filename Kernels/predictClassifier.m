function preds = predictClassifier(opt, X)
% make predictions
%  opt   - options
%     .theta    - (m+1)*K x 1 parameter vector.
%     .X        - training data. (only required when dual = true.)
%     .dual     - indicates whether using dual or not.
%     .kernelfn - kernel function - kernelfn(x, y) should return gram
%                 matrix between x and y.
%  X     - m x n design matrix.
%
%  preds - m x 1 vector of predicted y's.
%    
    m = size(X, 1);
    if opt.dual
        X = opt.kernelfn(X, opt.X);
    else
        X = [ones(m, 1), X];
    end
    n = size(X, 2);
    theta = reshape(opt.theta, n, []);
    [~, preds] = max(X*theta, [], 2); 
end
