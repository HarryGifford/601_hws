function opt = trainNN(X, y, opt)
%TRAINNN  Trains a neural network for classification problems.
%  X           - m x n design matrix
%  y           - m x 1 labels (must be discrete).
%                IMPORTANT: If there are k classes then the class labels
%                should be in 1:k.
%
%  opt - updated options struct containing learned parameters, theta.
%     .theta - learned NN parameters.
%     .hidden_sizes - numbers of hidden units in the neural network.
%

    % The following line converts the m x 1 vector of labels into an 
    % m x k indicator matrix, where the ith col will have k - 1 zeros and
    % 1 one in the jth row indicating that the ith observation is in
    % class j.
    %
    % e.g. if we get y = [1; 3; 1; 2] then we would get
    %      [1 0 1 0;
    %       0 0 0 1;
    %       0 1 0 0]
    y = full(sparse(1:length(y), y, 1));
    opt.output_size = size(y, 2);
    
    addpath(genpath('../shared'));
    
    % Set default parameters in case opt doesn't define them all.
    if ~isfield(opt, 'lambda'), opt.lambda = 0; end
    if ~isfield(opt, 'hidden_sizes'), opt.hidden_sizes = 64; end
    if ~isfield(opt, 'activation'), opt.activation = 'sigmoid'; end
    
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
    opt.theta = minFunc(@costNN, theta, opt, X, y, opt);
end
