function [cost, grad] = costNN(theta, X, y, opt)
% costNN  Neural network cost function.
%
% function [cost, grad] = costNN(X, y, theta, opt)
%
%   X     - m x n design matrix of m data points.
%   y     - m x k labels.
%   theta - flattened parameters for NN.
%   opt   - Struct must contain:
%          lambda        - regularization strength
%          hidden_sizes  - vector of number of units in each hidden layer. 
%                          In the case of a single hidden layer NN this 
%                          will be a scalar.
%
%   cost - cost at theta
%   grad - gradient at theta. i.e. [dJ/dx1, ..., dJ/dxp]'
%
    [m, visible_size] = size(X);
    hidden_size = opt.hidden_sizes;
    output_size = size(y, 2);
        
    n_layers = length(opt.hidden_sizes) + 1;

    all_layer_sizes = [visible_size; opt.hidden_sizes; output_size];
    
    [Ws, bs] = unflattenParameters(theta, all_layer_sizes);
    if visible_size == 28*28
        showImages(Ws{1}', 28, 28);
    end
    % You may find the following variables helpful.
    
    Wgrads = cell(n_layers, 1); % Wgrads{i} = Wigrad
    bgrads = cell(n_layers, 1); % bgrads{i} = bigrad
    
    % in the case of single hidden layer NN:
    %      layer 1 = input layer
    %      layer 2 = hidden layer
    %      layer 3 = output layer
    
    %% Write your code below to compute the cost and gradients.
    % our solution is ~20 lines. You may assume that the NN has a single
    % hidden layer.

    %% delete the following placeholder code when you start programming.
    % These three lines are here to prevent costNN crashing if you've not
    % started yet.
    cost = 0;
    Wgrads = Ws;
    bgrads = bs;
    
    % NOT YET IMPLEMENTED %
    
    %% BEGIN SOLUTION
    
    % forward propagation
    acts = cell(n_layers,1);
    acts{1}.a = X;
    for i = 1:n_layers-1
        z = bsxfun(@plus, acts{i}.a*Ws{i}, bs{i});
        if strcmp(opt.activation, 'sigmoid')
            [a, da] = sigmoid(z);
        elseif strcmp(opt.activation, 'softplus')
            [a, da] = softplus(z);
        elseif strcmp(opt.activation, 'tanh')
            a = tanh(2/3*z);
            da = 2/3*1.7159*(1 - a.^2);
            a = 1.7159*a;
        elseif strcmp(opt.activation, 'relu')
            a = max(0, z);
            da = z >= 0;
        end
        acts{i+1}.a = a;
        acts{i+1}.da = da;
    end
    
    % output layer cost
    hW = [bs{n_layers}; Ws{n_layers}];
    hA = [ones(m, 1), acts{n_layers}.a];
    [cost, grad, delta] = costMLR(hW, hA, y);

    % regularization cost.
    cost = cost + 1/(2*m)*opt.lambda*sum(cellfun(@(w) sum(w(:).^2), Ws));
    
    if nargout < 2, return; end % only need cost.
    grad = reshape(grad, size(hW));
    
    Wgrads{n_layers} = grad(2:end, :) + 1/m*opt.lambda*Ws{n_layers};
    bgrads{n_layers} = grad(1, :);
    % back propagation and grad computation
    for i = n_layers-1:-1:1
        delta = delta*Ws{i+1}'.*acts{i+1}.da;
        Wgrads{i} = acts{i}.a'*delta + 1/m*opt.lambda*Ws{i};
        bgrads{i} = sum(delta);
    end
    
    %% END SOLUTION
    
    % gradients should have same dimensionality as parameters. you can
    % delete this if you want.
    for i = 1:length(Ws)
        assert(all(size(Ws{i}) == size(Wgrads{i})));
        assert(all(size(bs{i}) == size(bgrads{i})));        
    end
    
    grad = flattenParameters(Wgrads, bgrads);

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
