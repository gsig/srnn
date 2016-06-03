function Y = rnn_softmaxlossb(Y,c,dzdY)
% RNN_SOFTMAXLOSSB  Backprop for Softmax Loss 
%   Y = rnn_softmaxlossb(Y,c,dzdY)
%   
%   Y: D x 1 vector
%   1 x 1 for c, the ground truth index
%
% Gunnar Atli Sigurdsson & Xinlei Chen 2015
% Carnegie Mellon University

if length(c) == 1
    Y(c) = Y(c) - 1;
else
    Y = Y - c;
Y = Y .* dzdY;
end
