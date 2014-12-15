function act = computeActivationsAE(theta, X, layer_idx, opt)
%COSTAE Weight tied autoencoder cost function.
%  theta - parameters
%  X - m x n design matrix.
%  layer_idx - layer to compute activations for.
%  opt - options
%     .num_hidden - number of hidden units.
%
    n = size(X, 2);
    W = reshape(theta(1:n*opt.num_hidden), n, opt.num_hidden);
    bvis_start = n*opt.num_hidden+1;
    bvis_end = bvis_start + n - 1;
    bvis = theta(bvis_start:bvis_end);
    bhid = theta(bvis_end+1:end);
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
    act = a2;
    if layer_idx == 1, return; end
    z3 = bsxfun(@plus, a2*W', bvis');
    if strcmp(opt.activation, 'sigmoid')
        act = sigmoid(z3);
    elseif strcmp(opt.activation, 'tanh')
        act = tanh(z3);
    else
        act = z3;
    end

end

function [y, dy] = sigmoid(x)
    y = 1./(1 + exp(-x));
    if nargout < 2, return; end
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
