function lcs = logsumexp(X, varargin)
% LOGSUMEXP(X, dim) computes log(sum(exp(X), dim)) robustly.
% Iain Murray, September 2010

    if (numel(varargin) > 1)
        error('Too many arguments')
    end

    if isempty(X)
        % Easiest way to get this trivial but annoying case right!
        lcs = log(sum(exp(X),varargin{:}));
        return;
    end

    if isempty(varargin)
        mx = max(X);
    else
        mx = max(X, [], varargin{:});
    end
    Xshift = bsxfun(@minus, X, mx);
    lcs = bsxfun(@plus, log(sum(exp(Xshift),varargin{:})), mx);

    idx = isinf(mx);
    lcs(idx) = mx(idx);
    lcs(any(isnan(X),varargin{:})) = NaN;
end