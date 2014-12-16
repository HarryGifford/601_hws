function runDigits()
% Trains a single layer NN on a subset of the MNIST set and evaluates the
% NN.
    addpath(genpath('../shared/'));
    data_path = '../data/mnist.mat';
    
    % don't change the parameters in this file.
    % we get ~88% accuracy with these parameters.
    opt.lambda = 10.0;
    opt.MaxIter = 1000; % max iterations for minimization function.
    
    opt = runLR(data_path, opt);
    
    % show the learned weights.
    load(data_path);
    
    showImages(opt.theta(2:end, :)', 28, 28);
    
    % show the misclassifications.
    misclassified = opt.test_preds(:) ~= y_test(:);
    if sum(misclassified) < 1600
        figure;
        showImages(X_test(misclassified,:),28,28);
    end

end