function X = rnn_cutoff(X,cut)
% RNN_CUTOFF  Clips X such that -cut<=X<=cut
%   
% Gunnar Atli Sigurdsson & Xinlei Chen 2015
% Carnegie Mellon University

X = min(max(X,-cut),cut);
end
