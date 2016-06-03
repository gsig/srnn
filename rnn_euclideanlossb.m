function Y = rnn_euclideanlossb(Y,c,dzdY)
% RNN_EUCLIDEANLOSSB  Calculates the euclidean loss of Y with respect to c
%   Y = RNN_EUCLIDEANLOSSB(Y,c,dzdY)
%   
%   D x 1 vector for Y
%   1 x 1 for c, the ground truth index
%   dzdY is the output gradient
%
% Gunnar Atli Sigurdsson & Xinlei Chen 2015
% Carnegie Mellon University

Y(:) = 2*(Y-c);
Y = Y * dzdY;
end
