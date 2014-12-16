function runDigits()
% Trains and tests an RBF SVM on a subset of the MNIST set.
    addpath(genpath('../shared/'));
    data_path = '../data/mnist.mat';
    
    % don't change the parameters in this file.
    % we get ~93.6% accuracy with these parameters.
    opt.lambda = 0.01;
    opt.dual = true;
    opt.gamma = 0.05;
    opt.MaxIter = 1000; % max iterations for minimization function.
    
    opt = runClassifier(data_path, opt);
    
    % show the learned weights.
    load(data_path);
        
    % show the misclassifications.
    misclassified = opt.test_preds(:) ~= y_test(:);
    if sum(misclassified) < 1600
        figure;
        showImages(X_test(misclassified,:),28,28);
    end

end