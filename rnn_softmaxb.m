function Y = rnn_softmaxb(Y,dzdY)
% RNN_SOFTMAXB  Backprop for Softmax nonlinearity
%   Y = rnn_softmaxb(Y,dzdY)
%   
%   Y: D x 1 vector
%
% Gunnar Atli Sigurdsson & Xinlei Chen 2015
% Carnegie Mellon University

Y = Y .* bsxfun(@minus, dzdY, sum(dzdY .* Y));
end
