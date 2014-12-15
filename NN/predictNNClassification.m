function preds = predictNNClassification(opt, X)
%NNPREDICTCLASSIFICATION Predict the classes of each x_i
%  opt - options
%     .theta        - neural network parameter vector.
%     .hidden_sizes - number of hidden units in hidden layer.
%  X   - m x n design matrix.
%
%  preds - 1 x m vector of predicted y's.
%    
    % hint: the following line might be helpful.
    probabilities = computeActivations(opt.theta, X, opt.output_size, opt);

    %% Compute the classes of each example in X.
    
    % NOT YET IMPLEMENTED %
    
    preds = zeros(size(X, 1), 1);
    
    %% BEGIN SOLUTION
    
    [~, preds] = max(probabilities, [], 2);

    %% END SOLUTION
    
end

function y = sigmoid(x)
    y = 1./(1. + exp(-x));
end