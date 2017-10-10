function y = vl_nnl2norm(x, param, varargin)
% VL_L2NORM normalizes the features to be norm 1 in 2-norm at each location
%
% Author: Subhransu Maji, Aruni RoyChowdhury, Tsung-Yu Lin

%
% This file is part of the BCNN and is made available under
% the terms of the BSD license (see the COPYING file).

% input:
% forward pass:
% x: input features of size [hight, width, channels, batches]
% param: the threshold to prevent large value when the norm is close to 0
% backward pass:
% x: input features of size [hight, width, channels, batches]
% param: the threshold to prevent large value when the norm is close to 0
% dzdy: the gradient with respect to output y

% output:
% forward pass:
% y: normalizing x to be norm 1.
% param: the threshold to prevent large value when the norm is close to 0
% backward pass:
% y: the gradient with respect to x

backMode = numel(varargin) > 0 && ~isstr(varargin{1}) ;
if backMode
  dzdy = varargin{1} ;
end

thresh = param(1);

gpuMode = isa(x, 'gpuArray');

[h, w, ch, bs] = size(x);
if gpuMode
    y = gpuArray(zeros([h, w, ch, bs], 'single'));
else
    y = zeros([h, w, ch, bs], 'single');
end

x_norm = sqrt(sum(x.*x, 3)+thresh);
if backMode
    E = bsxfun(@times, dzdy, x_norm.^(-1));
    F = sum(x.*dzdy,3);
    F = F.*x_norm.^(-3);
    F = bsxfun(@times, x, F);
    y = E-F;
else
    y = x./repmat(x_norm, [1, 1, ch, 1]);
end