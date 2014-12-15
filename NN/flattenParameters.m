function theta = flattenParameters(Ws, bs)
%FLATTENPARAMETERS Reverses the operation of unflatten
%   see also UNFLATTENPARAMETERS

ws = cellfun(@(x) x(:), Ws, 'UniformOutput', false);
w = cat(1, ws{:});
b = cat(2, bs{:});
theta = [w(:); b(:)];

end

