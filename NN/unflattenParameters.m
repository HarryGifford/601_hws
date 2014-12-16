function [Ws, bs] = unflattenParameters(theta, layer_sizes)
%function [Ws, bs] = unflattenParameters(theta, layer_sizes)
%  Takes a 1d vector of NN parameters and turns it into a vector of weights
%  and biases.
%
%  theta       - q x 1 vector of parameters.
%  layer_sizes - (L+1) x 1 vector in the form [input_size; hidden_sizes;
%                                              output_size]
%
%  Ws - Ws{i} is an insize x outsize parameter matrix of weights.
%  bs - bs{i} is an insize x 1 parameter vector of bias units.
%
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
