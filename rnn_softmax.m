function Y = rnn_softmax(X)
% RNN_SOFTMAX  Softmax nonlinearity
%   Y = rnn_softmax(X)
%   
%   X: D x 1 vector
%
% Gunnar Atli Sigurdsson & Xinlei Chen 2015
% Carnegie Mellon University

E = exp(bsxfun(@minus, X, max(X)));
L = sum(E);
Y = bsxfun(@rdivide, E, L);
end
