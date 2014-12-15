function opt = trainNN(X, y, opt)
%TRAINNN  Trains a neural network with opt.hidden_sizes hidden units.
%  X   - m x n design matrix
%  y   - m x k outputs for each x_i
%  opt - Struct containing NN parameters.
%     .lambda - regularization strength.
%     .hidden_sizes - number of hidden units in the hidden layer(s).
%
%  opt - Updated struct containing NN parameters and learned theta.
%     .theta - flattened vector of ALL parameters in the neural network.
% 
    addpath(genpath('../shared'));
    
    % Set default parameters in case opt doesn't define them all.
    if ~isfield(opt, 'lambda')
        opt.lambda = 0;
    end
    if ~isfield(opt, 'hidden_sizes')
        opt.hidden_sizes = 64;
    end
    if ~isfield(opt, 'activation')
        opt.activation = 'sigmoid';
    end
    
    all_layer_sizes = [size(X, 2); opt.hidden_sizes; size(y, 2)];
    
    if isfield(opt,'init_theta')
        theta = opt.init_theta;
    else
        [Ws, bs] = initializeParameters(all_layer_sizes);
        theta = flattenParameters(Ws, bs);
    end
    
    % check that the gradients look reasonable.
    toyopt.hidden_sizes = 4;
    toyopt.lambda = 2.0;
    toyopt.activation = opt.activation;
    [toyws, toybs] = initializeParameters([3; 4; 2]);
    toyw = flattenParameters(toyws, toybs);
    toyopt.init_theta = toyw;
    toyX = rand(10, 3);
    toyY = rand(10, 1) < 0.5;
    toyY = [toyY, ~toyY];
    checkGradient(@(t) costNN(t, toyX, toyY, toyopt), toyw);

    % For this part of the assignment we will use a minimization
    % tool, minFunc. The interface is similar to the gradient descent
    % optimizer you wrote, but it uses some fancy tricks to try and
    % approximate the Hessian in order to reduce the number of times J must
    % be called.
    
    % type the following into the matlab terminal to compile minFunc:
    % >> cd ../shared/
    % >> addpath ./minFunc/
    % >> mexAll
    addpath(genpath('minFunc/'));
    
    opt.theta = minFunc(@(t) costNN(t, X, y, opt), theta, opt);
end
