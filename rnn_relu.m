function Y = rnn_relu(X,cut)
% RNN_RELU  Rectified Linear Unit
%   Y = rnn_relu(X) rectifies X
%   Y = rnn_relu(X,cut) clips Y such that -cut<=Y<=cut
%   
%   X is a D x 1 vector
%   
% Gunnar Atli Sigurdsson & Xinlei Chen 2015
% Carnegie Mellon University

if nargin == 2
	X = min(max(X,-cut),cut);
end
Y = max(X,0);
end
