function theta = minimize(f, theta, X, y, opt)
% MINIMIZE  Find a local minima of a function f, starting at theta.
%          f - function to be minimized. f should be of the form:
%              [cost, grad] = f(theta, X, y)

    if nargin < 5 % no options provided.
        opt.tol = 1e-6; % you should stop optimization when the absolute
                    % difference in cost between two iterations is less
                    % than tol.  
        opt.maxIter = 1000; % you should also break after maxIter if you
                             % haven't reached tol yet.
        opt.alpha = 2;
        opt.gamma = 0.7;
        opt.alpha_decay = 0.998;
        opt.batch_size = 500;
    end
    
    m = size(X, 1);
    % Write your solution below. You should use gradient descent to find
    % a minimum of the function.
    % Our solution is ~10 lines
    
    %% BEGIN SOLUTION (GRADIENT DESCENT)
    
    prev_cost = inf;
    alpha = opt.alpha;
    v = zeros(size(theta));
    fprintf('Running Gradient Descent...\n');
    for iter = 1:opt.maxIter
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
        if mod(iter, 1) == 0,
            fprintf('Running epoch %d/%d with cost = %.5f\n',...
                    iter, opt.maxIter, cost);
        end
        prev_cost = cost;
    end 
    fprintf('Converged in %d iterations with cost = %.5f!\n', iter, cost);
  
    %% END SOLUTION
end
