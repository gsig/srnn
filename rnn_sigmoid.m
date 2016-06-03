function Y = rnn_sigmoid(X,cut)
% RNN_SIGMOID  Sigmoid nonlinearity 
%   Y = rnn_sigmoid(X)
%   Y = rnn_sigmoid(X,cut) Y is clipped between -cut and cut
%   
%   X: D x 1 vector
%
% Gunnar Atli Sigurdsson & Xinlei Chen 2015
% Carnegie Mellon University

if nargin == 2
	X = min(max(X,-cut),cut);
end
Y = 1 ./ (1 + exp(-X));
end
