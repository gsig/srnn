function net = rnn_updatenet(net, alpha, momentum, n)
% RNN_UPDATENET  Uses previously calculated gradients to update the network
% Implements stochastic gradient descent with momentum, n is the batch size, alpha learning rate
% The gradients are calculated for net by calling RNN_BACKWARD, see RNN_TRAINNET
%   net = rnn_updatenet(net, alpha, momentum, n)
%   
% Gunnar Atli Sigurdsson & Xinlei Chen 2015
% Carnegie Mellon University


% input
l = length(net.layers_in);
for i=1:l
	% batch size 
	net.grad_in{i} = net.grad_in{i} / n;
	net.momentum_in{i} = momentum * net.momentum_in{i} - net.regu_in(i) * net.layers_in{i} ...
		- net.grad_in{i};
	net.layers_in{i} = net.layers_in{i} + alpha * net.momentum_in{i};
end

% self
l = length(net.layers_self);
for i=1:l
	% batch size 
	net.grad_self{i} = net.grad_self{i} / n;
	net.momentum_self{i} = momentum * net.momentum_self{i} - net.regu_self(i) * net.layers_self{i} ...
		- net.grad_self{i};
	net.layers_self{i} = net.layers_self{i} + alpha * net.momentum_self{i};
end

% output
l = length(net.layers_out);
for i=1:l
	% batch size 
	net.grad_out{i} = net.grad_out{i} / n;
	net.momentum_out{i} = momentum * net.momentum_out{i} - net.regu_out(i) * net.layers_out{i} ...
		- net.grad_out{i};
	net.layers_out{i} = net.layers_out{i} + alpha * net.momentum_out{i};
end

end
