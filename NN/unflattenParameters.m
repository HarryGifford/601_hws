%function [W1, W2, b1, b2] = unflattenParameters(theta, output_size, ...
%                                                hidden_size, visible_size)
%UNFLATTENPARAMETERS  convert theta vector into different NN parameters.
%  Takes the really long theta parameter vector used for optimization and 
%  converts it into the weight and bias matrices.
%
%  theta - vector of all NN parameters
%
%  W1 - Weights of connections between neurons in the input and hidden
%       layers.
%  W2 - Same as above for hidden and output layers.
%  b1 - biases for the hidden layer.
%  b2 - biases for the output layer.
%
%
%    [Ws, bs] = unflattenParameters_(theta, [visible_size; hidden_size; output_size]);
%    W1 = Ws{1};
%    W2 = Ws{2};
%    b1 = bs{1};
%    b2 = bs{2};
%end

function [Ws, bs] = unflattenParameters(theta, layer_sizes)
    n_layers = length(layer_sizes);
    Ws = cell(n_layers-1, 1);
    bs = cell(n_layers-1, 1);
    W_offset = 1;
    b_offset = 1+dot(layer_sizes(1:end-1), layer_sizes(2:end));
    for i = 1:(n_layers-1)
        in_size = layer_sizes(i);
        out_size = layer_sizes(i+1);
        W_offset_new = W_offset + in_size*out_size;
        b_offset_new = b_offset + out_size;
        Ws{i} = reshape(theta(W_offset:W_offset_new-1), in_size, out_size);
        bs{i} = theta(b_offset:b_offset_new-1)';
        W_offset = W_offset_new;
        b_offset = b_offset_new;
    end
end