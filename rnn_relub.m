function Y = rnn_relub(Y,dzdY)
% RNN_RELUB  Backprop for Rectified Linear Unit
%   Y = rnn_relub(Y,dzdY)
%   
%   Y: D x 1 vector
%
% Gunnar Atli Sigurdsson & Xinlei Chen 2015
% Carnegie Mellon University

Y = dzdY .* (Y > 0) ;
end
