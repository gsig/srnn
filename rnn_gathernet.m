function net = rnn_gathernet(net)
% RNN_GATHERNET  Transfers data from the gpu to the cpu, only useful with gpus
%   net = RNN_GATHERNET(net)
%   
% Gunnar Atli Sigurdsson & Xinlei Chen 2015
% Carnegie Mellon University


% in
l = length(net.layers_in);
for i=1:l
	net.layers_in{i} = gather(net.layers_in{i});
	net.grad_in{i} = gather(net.grad_in{i});
	net.momentum_in{i} = gather(net.momentum_in{i});
end
net.regu_in = gather(net.regu_in);

% self
l = length(net.layers_self);
for i=1:l
	net.layers_self{i} = gather(net.layers_self{i});
	net.grad_self{i} = gather(net.grad_self{i});
	net.momentum_self{i} = gather(net.momentum_self{i});
end
net.regu_self = gather(net.regu_self);

% out
l = length(net.layers_out);
for i=1:l
	net.layers_out{i} = gather(net.layers_out{i});
	net.grad_out{i} = gather(net.grad_out{i});
	net.momentum_out{i} = gather(net.momentum_out{i});
end
net.regu_out = gather(net.regu_out);

end
