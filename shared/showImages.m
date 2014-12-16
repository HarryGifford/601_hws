function showImages(X, xdim, ydim)
%SHOWIMAGES Displays grayscale or color images in grid.
%  X    - m x xdim*ydim*n_channels matrix of images
%  xdim - number of columns
%  ydim - (optional) number of rows. If not provided, assumes xdim = ydim.
%
%  Modified from 
%   Quoc V. Le, On Optimization Methods for Deep Learning, ICML 2011
%
    %figure;
    if (nargin < 3), ydim = xdim; end
    
    border = 2;

    cma = max(X(:));
    cmi = min(X(:));

    X = (X - cmi)./(cma - cmi);
    
    [m, dim] = size(X);
    n_channels = dim/(xdim*ydim);
    assert(n_channels == 1 || n_channels == 3);
    X = reshape(X, m, ydim, xdim, n_channels);
    rootm = ceil(sqrt(m));        
    grid = ones(rootm*(xdim + border) + border, ...
        rootm*(ydim + border) + border, n_channels);
    for y = 1:rootm
        for x = 1:rootm
            idx = y*rootm + x - rootm;
            if idx > m, break; end
            y_start = border + (y-1)*(ydim + border) + 1;
            y_end = y_start + ydim-1;
            x_start = border + (x-1)*(xdim + border) + 1;
            x_end = x_start + xdim-1;
            grid(y_start:y_end, x_start:x_end,:) = X(idx, :, :, :);
        end
    end
    if isOctave()
        imshow(grid);drawnow;drawnow;
    else
        imshow(grid, 'InitialMagnification', 100);drawnow;drawnow;
    end
end

