function act = computeActivations(theta, X, output_size, opt)
%NNCOMPUTEACTIVATIONS  Compute the activations from the last layer of the
%                     neural network.
%
% function act = nnComputeActivations(theta, X, output_size, opt)
%
% theta       - learned parameter vector of all weights in the NN.
% X           - m x n design matrix. Note that in this case we will not
%               add a bias vector to the X's directly, since we are not
%               going to regularize the biases.
% layer_sizes - size of hidden layers and output layer.
% opt         - NN options
%
% act - activations of the output layer.
%
% Example usage:
%  acts = nnComputeActivations(theta, X, layers, opt);
%  acts = nnComputeActivations(theta, X, [hidsizes; outsize], opt);
%
    [m, visible_size] = size(X);
    layer_sizes = [visible_size; opt.hidden_sizes; output_size];
    
    [Ws, bs] = unflattenParameters(theta, layer_sizes);
        
    %% Compute the activations of the output layer. Our solution is approx 
    %  10 lines.
    
    % NOT YET IMPLEMENTED %

    act = zeros(output_size, m);

    %% BEGIN SOLUTION
    act = X;
    for i = 1:length(layer_sizes)-1
        z = bsxfun(@plus, act*Ws{i}, bs{i});
        if strcmp(opt.activation, 'sigmoid')
            act = sigmoid(z);
        elseif strcmp(opt.activation, 'softplus')
            act = softplus(z);
        elseif strcmp(opt.activation, 'tanh')
            act = 1.7159*tanh(z);
        elseif strcmp(opt.activation, 'relu')
            act = max(0, z);
        end
    end
    %% END SOLUTION
end


function [y, dy] = sigmoid(x)
    y = 1./(1 + exp(-x));
    if nargout < 2, return; end
    dy = y.*(1-y);
end

function [y, dy] = softplus(x)
    y = log1p(exp(x)) + 0.01*x;
    if nargout < 2, return; end
    dy = sigmoid(x) + 0.01;
end

function [y, dy] = tnh(x)
% See Efficient Backprop by Y LeCun for more info on where the magic
% constants come from.
    y = tanh(2/3*x);
    if nargout >= 2
        dy = 2/3*1.7159*(1 - y.^2) + 0.01;
    end
    y = 1.7159*y + 0.01*x;
end
