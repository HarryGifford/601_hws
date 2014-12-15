function opt = runNN(dataset, opt)
% fits a neural network to a non-linearly seperable 2D binary dataset and
% then plots the estimated probabilities.
%
% dataset - either 'random' or './path/to/dataset/' containing
%           entries X_train, X_test, y_train, (y_test - optional).
% opt     - options to run with:
%    .lambda - regularization strength.
%    .hidden_sizes - number of units in hidden layer.
%
% opt     - updated options.
%    .theta - learned parameters.
%    .test_preds - test set predictions.
%
    if nargin < 1, dataset = 'random'; end
    if nargin < 2
        opt.lambda = 0.01;
        opt.hidden_sizes = 20;
        opt.display = true;
        opt.MaxIter = 400;
    end
    if ~isfield(opt, 'lambda'), opt.lambda = 0.01; end
    if ~isfield(opt, 'hidden_sizes'), opt.hidden_sizes = 20; end
    if ~isfield(opt, 'display'), opt.display = true; end
    
    addpath(genpath('../shared/'));

    if strcmp(dataset, 'random')
        n = 2; % dimensionality of data to generate.
        K = 3; % number of classes to generate.
        centers = 2*rand(K, n)-1;
        [X_train, y_train] = generateData(100, centers);
        [X_test, y_test] = generateData(100, centers);
        ymin = min(y_train(:));
        y_train = y_train - ymin + 1;
    else
        load(dataset);
        n = size(X_train, 2);
        y_train = double(y_train);
        ymin = min(y_train(:));
        y_train = y_train - ymin + 1;
        if ~exist('y_test', 'var')
            y_test = -ones(size(X_test, 1), 1); % dummy test labels.
        else
            y_test = double(y_test);
            y_test = y_test - ymin + 1;
        end
        K = max(y_train(:));
    end
    
    % train and test classifier
    opt = trainNNClassification(X_train(1:end,:), y_train(1:end), opt);
    preds = predictNNClassification(opt, X_train);
    fprintf('Train Accuracy = %.2f%%\n', 100*mean(preds(:) == y_train(:)));
    
    preds = predictNNClassification(opt, X_test);
    fprintf('Test Accuracy = %.2f%%\n', 100*mean(preds(:) == y_test(:)));    
    
    opt.test_preds = preds + ymin - 1;
    
    if n ~= 2 || ~isfield(opt, 'display') || ~opt.display, return; end;
    
    xmin = min(X_train(:, 1));
    xmax = max(X_train(:, 1));
    ymin = min(X_train(:, 2));
    ymax = max(X_train(:, 2));
    
    if K <= 6 % make pretty colors for scatter plot for up to 6 classes.
        colors = [0 0 1; 1 0 0; 0 1 0; 0 1 1; 1 1 0; 1 0 1];
        light_colors = colors + .8;
        light_colors(light_colors > 1) = 1;
        colormap(light_colors(1:K, :));
    end
    

    
    hold on;
    plotBoundary(@(x) predictNNClassification(opt, x),...
                 xmin, xmax, ymin, ymax);
    if K <= 6, cy = colors(y_train,:); else cy = y_train; end;
    scatter(X_train(:, 1), X_train(:, 2), [], cy, 'filled');
    axis equal tight
    hold off;
    
    return;

end

function [X, y] = generateData(m, centers)
% Generates some random guassian data.
%  m - number of data points to generate per class
%  centers - K x n matrix of cluster centers
% 
%  X - m*K x n design matrix
%  y - m*K x 1 labels
%
    if numel(centers) == 1, centers = randn(centers, 2); end

    [K, n] = size(centers);
    X = zeros(m*K, n);
    y = zeros(m*K, 1);
    for i = 1:K
        start_idx = (i-1)*m+1;
        end_idx = i*m;
        data = bsxfun(@plus, 0.2*randn(m, n), centers(i, :));
        X(start_idx:end_idx, :) = data;
        y(start_idx:end_idx) = i;
    end
end

function plotBoundary(predictClass, xmin, xmax, ymin, ymax)
% Plots decision boundary of this classifier.
%  predictClass - function that takes a set of data and predicts class
%                 label.
%  xmin, xmax, ymin, ymax - bounds to plot.
%
    if nargin < 4, ymin = xmin; ymax = xmax; end

    xrnge = linspace(xmin, xmax, 300);
    yrnge = linspace(ymin, ymax, 300);

    [xs, ys] = meshgrid(xrnge, yrnge);
    X = [xs(:), ys(:)];
    y = predictClass(X);
    y = reshape(y, size(xs));
    K = max(y(:));
    if isOctave()
        contour(xs, ys, (y-1)./(K-1), K-1, 'k-', 'LineWidth', 2);
    else
        contourf(xs, ys, (y-1)./(K-1), K-1, 'k-', 'LineWidth', 1);
    end
end