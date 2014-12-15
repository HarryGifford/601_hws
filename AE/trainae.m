function [theta, W, bhidden, bvis] = trainae(X, opt)
%TRAINAE Trains a simple weight tied autoencoder
%  X   - m x n design matrix
%  opt - options
%     .lambda - regularization
%     .num_hidden - number of hidden units
%     .noise - corruption level
%
    [m, n] = size(X);
    W = sqrt(6/(opt.num_hidden+n))*(2*rand(n, opt.num_hidden)-1);
    bvis = zeros(1, n);
    bhidden = zeros(1, opt.num_hidden);
    
    theta = [W(:); bvis(:); bhidden(:)];
    iter = ceil(opt.MaxIter)/10;
    BATCH_SIZE = min(400, m);
    for i = 1:iter
        opt.MaxIter = 4;
        for j = 1:BATCH_SIZE:m
            fprintf('Running iteration %d/%d, batch %d/%d\n', i,...
                iter, round(j/BATCH_SIZE), round(m/BATCH_SIZE)); 
            opt.Display = false;
            X_ = X(j:min(j+BATCH_SIZE-1, m),:);
            msk = rand(size(X_)) >= opt.noise;
            if strcmp(opt.activation, 'tanh')
                Xm = 2*msk.*X_-1;
            else
                Xm = msk.*X_;
            end
            theta = minFunc(@costAE, theta(:), opt, Xm, X_, opt);
        end
    end
    if nargout < 2, return; end
    
    W = reshape(theta(1:n*opt.num_hidden), n, opt.num_hidden);
    bvis_start = n*opt.num_hidden+1;
    bvis_end = bvis_start + n - 1;
    bvis = theta(bvis_start:bvis_end);
    bhidden = theta(bvis_end+1:end);    
end

