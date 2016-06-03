function Y = rnn_sigmoidb(Y,dzdY)
% RNN_SIGMOIDB Backprop for Sigmoid nonlinearity 
%   Y = rnn_sigmoidb(Y,dzdY)
%   
%   Y: D x 1 vector
%
% Gunnar Atli Sigurdsson & Xinlei Chen 2015
% Carnegie Mellon University

Y = dzdY .* (Y .* (1 - Y)) ;
end
