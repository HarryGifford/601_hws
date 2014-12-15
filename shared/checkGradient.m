function checkGradient(costfn, theta)
% CHECKGRADIENT  Checks your gradient against finite differences gradient
%               computed from function evaluation. Prints a warning if
%               L2 norm difference (sum of squares of differences) is
%               too large.
%
%   J should be a function of the form
%        [cost : scalar, grad : mx1 vector] = J(theta : mx1 vector)
%
    tol = 1e-6;
    n_dims = size(theta, 1);
    dx = 1e-5;
    id = eye(n_dims)*dx;
    grad_hat = zeros(size(theta));
    [~, grad] = costfn(theta); % get true analytical gradient.
    for i = 1:n_dims % now compute finite differences gradient.
        xp1 = costfn(theta + id(:, i));
        xm1 = costfn(theta - id(:, i));
        grad_hat(i) = (xp1 - xm1)/(2*dx); 
    end
    diff = norm(grad_hat - grad);
    
    if diff > tol
        warning(['Your gradients differ by %f. Your gradient'...
                 ' or cost function may be incorrect.\n'], diff);
        fprintf('Numeric grad Your grad\n');
        disp([grad_hat, grad]);
        pause(5);
    end
end
