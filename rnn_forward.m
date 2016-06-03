function [hidden, output] = rnn_forward(input, hidden, net, actCut)
% RNN_FORWARD  Forward pass of the RNN 
%   [hidden, output] = RNN_FORWARD(input, hidden, net, actCut)
%   
%   input: cell that contains 1 x D vectors
%   hidden: cell that contains hidden states as 1 x H vectors
%   output: cell that contains 1 x D vectors
%   
% Gunnar Atli Sigurdsson & Xinlei Chen 2015
% Carnegie Mellon University


% hidden
hidden{1} = input{1} * net.layers_in{1} + hidden{1} * net.layers_self{1};

% sigmoid
hidden{1} = rnn_sigmoid(hidden{1},actCut);

% hidden to output
output = cell(1,1);
output{1} = hidden{1} * net.layers_out{1};

% softmax
output{1} = rnn_relu(output{1});

end
