function preds = predictLR(opt, X)
%PREDICTLR Predict class labels for X.
%  opt - options
%     .theta        - (n+1)*k x 1 parameter vector.
%  X   - m x n design matrix.
%
%  preds - 1 x m vector of predicted y's.
%    

    [m, n] = size(X);
    theta = reshape(opt.theta, n+1, []);
    
    %% BEGIN SOLUTION
    
    X = [ones(m, 1), X];
    [~, preds] = max(X*theta, [], 2);
    
    %% END SOLUTION

    preds = preds + opt.ymin - 1;
    
end

