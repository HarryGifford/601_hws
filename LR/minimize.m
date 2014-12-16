function theta = minimize(f, theta, X, y, opt)
% MINIMIZE  Find a local minima of a function f, starting at theta.
%  f     - function to be minimized. f should be of the form:
%              [cost, grad] = f(theta, X, y)
%  theta - initial parameters.
%  X     - m x n design matrix.
%  y     - m x k target matrix.
%  opt   - options
%     .tol         - you should stop optimization when the absolute
%                    difference in cost between two iterations is less
%                    than tol.
%     .MaxIter     - Alternatively, break after reaching this number of
%                    iterations over the data.
%     .alpha       - Gradient step size.
%     .alpha_decay - Decay alpha by this constant after each pass over the
%                    data.
% 
%  theta - optimized paramaters.
% 
    if nargin < 5 % no options provided.
        opt = struct();
    end
    % set defaults for options that are not provided.
    if ~isfield(opt, 'tol'), opt.tol = 1e-6; end
    if ~isfield(opt, 'MaxIter'), opt.MaxIter = 1000; end
    if ~isfield(opt, 'alpha'), opt.alpha = 2; end
    if ~isfield(opt, 'gamma'), opt.gamma = 0.7; end
    if ~isfield(opt, 'alpha_decay'), opt.alpha_decay = 0.998; end
    if ~isfield(opt, 'batch_size'), opt.batch_size = 500; end
    
    m = size(X, 1);
    
    % Write your solution below. You should use gradient descent to find
    % a minimum of the function.
    % Our solution is ~10 lines
    
    %% BEGIN SOLUTION (GRADIENT DESCENT)
    
    prev_cost = inf;
    alpha = opt.alpha;
    v = zeros(size(theta));
    fprintf('Running Gradient Descent...\n');
    for iter = 1:opt.MaxIter
        idxs = randperm(m);
        X = X(idxs, :);
        y = y(idxs, :);
        for start_idx = 1:opt.batch_size:m            
            end_idx = min(m, start_idx + opt.batch_size-1);
            [cost, grad] = f(theta, X(start_idx:end_idx,:), y(start_idx:end_idx,:));
            if isnan(cost) || cost > 1e30
                prev_cost = inf;
                theta = 0.01*randn(size(theta));
                alpha = 0.5*alpha;
                continue;
            end
            v = opt.gamma*v + alpha*grad;
            theta = theta - v;
            alpha = alpha*opt.alpha_decay;
        end
        cost = f(theta, X, y);
        if abs(cost - prev_cost) < opt.tol, break; end
        if mod(iter, 10) == 0,
            fprintf('Running epoch %d/%d with cost = %.5f\n',...
                    iter, opt.MaxIter, cost);
        end
        prev_cost = cost;
    end 
    fprintf('Converged in %d iterations with cost = %.5f!\n', iter, cost);
  
    %% END SOLUTION
end
