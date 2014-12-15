function [cost, grad] = costAE(theta, X, y, opt)
%COSTAE Weight tied autoencoder cost function.
%  X - m x n design matrix.
%  y - m x n target matrix.
%  opt - options
%     .lambda - regularization strength.
%     .num_hidden - number of hidden units.
%     .p - target sparsity
%     .beta - sparsity strength
%

    p = opt.p;
    [m, n] = size(X);
    W = reshape(theta(1:n*opt.num_hidden), n, opt.num_hidden);
    bvis_start = n*opt.num_hidden+1;
    bvis_end = bvis_start + n - 1;
    bvis = theta(bvis_start:bvis_end);
    bhid = theta(bvis_end+1:end);
        if n == 28*28
            showImages(W', 28);
        end
    z2 = bsxfun(@plus, X*W, bhid');
    if strcmp(opt.hidden_activation, 'sigmoid')
        [a2, d2] = sigmoid(z2);
    elseif strcmp(opt.hidden_activation, 'softplus')
        [a2, d2] = softplus(z2);
    elseif strcmp(opt.hidden_activation, 'tanh')
        a = tanh(2/3*z2);
        d2 = 2/3*1.7159*(1 - a.^2);
        a2 = 1.7159*a;
    elseif strcmp(opt.hidden_activation, 'relu')
        a2 = max(0, z2);
        d2 = z2 >= 0;
    else
        a2 = z2;
        d2 = ones(size(a2));
    end
        
    z3 = bsxfun(@plus, a2*W', bvis');
    if strcmp(opt.activation, 'sigmoid')
        a3 = sigmoid(z3);
        d3 = a3.*(1-a3);
    elseif strcmp(opt.activation, 'tanh')
        a3 = tanh(z3);
        d3 = 1 - a3.^2;
    else
        a3 = z3;
        d3 = ones(size(z3));
    end
    
    if opt.beta > 0
        %p_hats = mean(a2);
        %kl = sum(p*(log(p) - log(p_hats)) + (1-p)*(log(1-p) - log(1-p_hats)));
        kl = opt.beta./m*sum(sqrt(a2(:).^2 + 1));
    else
        kl = 0;
    end
    diff = a3 - y;
    if false%strcmp(opt.activation, 'sigmoid')
        cost = 1/m*(sum(log1p(exp(z3(:)))) - y(:)'*z3(:));
        delta3 = diff;
    else
        cost = 1/(2*m)*sum(diff(:).^2);
        delta3 = diff.*d3;
    end
    %cost = 1/m*(sum(log1p(exp(-y(:)'*z3(:)))));
    cost = cost + 1/(2*m)*opt.lambda*sum(W(:).^2) + kl;
    if nargout < 2, return; end

    if opt.beta > 0
        sparse_term = opt.beta./m*bsxfun(@rdivide, a2, sum(sqrt(1 + a2.^2)));
        %sparse_term = opt.beta*((1-p)./(1-p_hats) - p./p_hats);
        %size(sparse_term)
        %dfas
    else
        sparse_term = 0;
    end
    delta2 = bsxfun(@plus, delta3*W.*d2,sparse_term);
    WWgrad = 1./m*a2'*delta3;
    bvisgrad = 1./m*sum(delta3)';

    Wgrad = 1./m*X'*delta2 + 1./m*opt.lambda*W;
    bhidgrad = 1./m*sum(delta2)';

    WWgrad = WWgrad';
    grad = [Wgrad(:) + WWgrad(:); bvisgrad; bhidgrad];

end

function [y, dy] = sigmoid(x)
    y = 1./(1 + exp(-x));
    dy = y.*(1-y);
end

function [y, dy] = softplus(x)
    y = log1p(exp(x));
    dy = sigmoid(x);
end

function [y, dy] = linear(x)
    y = x;
    dy = ones(size(x));
end
