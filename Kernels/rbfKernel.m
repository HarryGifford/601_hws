function K = rbfKernel(x, y, gamma)
    K = -2*x*y';
    K = bsxfun(@plus, K, sum(x.^2,2));
    K = bsxfun(@plus, K, sum(y.^2,2)');
	K = exp(-gamma*K);
end
