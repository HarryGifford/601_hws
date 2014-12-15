function K = polyKernel(x, y, order)
    K = (1+x*y').^order;
end
