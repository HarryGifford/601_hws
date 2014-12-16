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
    if ~isfield(opt, 'MaxIter'), opt.MaxIter = 6000; end
    if ~isfield(opt, 'alpha'), opt.alpha = 3; end
    if ~isfield(opt, 'alpha_decay'), opt.alpha_decay = 0.998; end
    
    m = size(X, 1);
    
    % Write your solution below. You should use gradient descent to find
    % a minimum of the function.
    % Our solution is ~10 lines
    
    %% BEGIN SOLUTION (GRADIENT DESCENT)
    
    prev_cost = inf;
    alpha = opt.alpha;
    fprintf('Running Gradient Descent...\n');
    for iter = 1:opt.MaxIter
        [cost, grad] = f(theta, X, y);
        theta = theta - alpha*grad;
        if abs(cost - prev_cost) < opt.tol, break; end
        if mod(iter, 10) == 0,
            fprintf('Running iteration %d/%d with cost = %.5f\n',...
                    iter, opt.MaxIter, cost);
        end
        prev_cost = cost;
        alpha = alpha*opt.alpha_decay;
    end 
    fprintf('Converged in %d iterations with cost = %.5f!\n', iter, cost);
  
    %% END SOLUTION
end
