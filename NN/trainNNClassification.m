function opt = trainNNClassification(X, y, opt)
%  Wrapper function for trainNN for classification problems.
%  Converts the m x 1 vector of labels into an m x k indicator matrix,
%  where the ith col will have k - 1 zeros and 1 one in the jth row
%  indicating that the ith observation is in class j. This indicator matrix
%  is then passed to nnTrain.
%
%  e.g. if we get y = [1; 3; 1; 2] then we would get an indicator matrix of
%       [1 0 1 0;
%        0 0 0 1;
%        0 1 0 0]
%
%  X           - m x n design matrix
%  y           - m x 1 labels (must be discrete).
%                IMPORTANT: If there are k classes then the class labels
%                should be in 1:k.
%  hidden_size - number of hidden units in the neural network.
%  output_size - number of units in output layer. Should be the number of
%                classes.
%
%  opt - updated options struct containing learned parameters, theta.
%     .theta - learned NN parameters.
% 

    y_train = full(sparse(1:length(y), y, 1));
    opt.output_size = size(y_train, 2);
    opt = trainNN(X, y_train, opt);
end
