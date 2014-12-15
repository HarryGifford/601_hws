function runDigits()
% Trains a single layer NN on a subset of the MNIST set and evaluates the
% NN.
    addpath(genpath('../shared/'));
    data_path = '../data/mnist.mat';
    
    % don't change the parameters in this file.
    % we get ~92% accuracy with these parameters.
    opt.hidden_sizes = 64;
    opt.lambda = 1e-2;
    opt.MaxIter = 400; % max iterations for minimization function.
    
    opt = runNN(data_path, opt); % train and test NN.
    
    % show the learned weights.
    load(data_path);
    
    visible_size = size(X_train, 2);
    K = max(y_train);
    
    Ws = unflattenParameters(opt.theta, [visible_size; opt.hidden_sizes; K]);
    showImages(Ws{1}', 28, 28);
    
    % show the misclassifications.
    misclassified = opt.test_preds(:) ~= y_test(:);
    if sum(misclassified) < 1600
        figure;
        showImages(X_test(misclassified,:),28,28);
    end

end