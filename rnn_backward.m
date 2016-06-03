function [ghidden, net] = rnn_backward(input, hidden, output, gt, ghidden_in, net, gradCut, outputgrad)
% RNN_BACKWARD  Calculates the gradients for all nodes of the RNN
% [ghidden, net] = RNN_BACKWARD(input, hidden, output, gt, ghidden_in, net, gradCut, outputgrad)
%
%   ghidden is the gradient for the hidden layer
%   
% Gunnar Atli Sigurdsson & Xinlei Chen 2015
% Carnegie Mellon University


% grad for output
goutput = cell(1,1);
%goutput{1} = rnn_cutoff(rnn_softmaxlossb(output{1},gt,1),gradCut); 
%goutput{1} = rnn_cutoff(rnn_euclideanlossb(output{1},gt,1),gradCut); 
goutput{1} = rnn_cutoff(rnn_relub(output{1},outputgrad),gradCut); 

% grad for output layer
net.grad_out{1} = net.grad_out{1} + hidden{1}' * goutput{1};

% grad for hidden layer
net.grad_self{1} = net.grad_self{1} + hidden{1}' * ghidden_in{1};

% grad for hidden
ghidden = cell(1,1);
ghidden{1} = rnn_cutoff(ghidden_in{1} + goutput{1} * net.layers_out{1}',gradCut);
ghidden{1} = rnn_sigmoidb(hidden{1}, ghidden{1});

% grad for the input layer
net.grad_in{1} = net.grad_in{1} + input{1}' * ghidden{1};

end
