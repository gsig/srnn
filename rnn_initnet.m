function net = rnn_initnet(varargin)
% RNN_INITNET  Initilize the rnn model
%   net = rnn_initnet(varargin)
%   
% Gunnar Atli Sigurdsson & Xinlei Chen 2015
% Carnegie Mellon University

opts.scale = 1;
opts.weightDecay = 0.0000001;
opts.gpu = 0;
opts.hidden_size = 10;
opts.input_size = 101;
opts.max_val = 36.68;
opts.repeatTrain = 5;
opts.subsampl = 12;
opts = vl_argparse(opts, varargin) ;
opts

net = [];
net.opts = opts;
% Three part of the weights
% Input
net.layers_in = {};
net.regu_in = [];
net.grad_in = {};
net.momentum_in = {};
in = opts.input_size;
out = opts.hidden_size;
net.layers_in{1} = 0.01/opts.scale * randn(in, out, 'single');
net.regu_in(1) = single(opts.weightDecay);
net.grad_in{1} = zeros(in, out, 'single');
net.momentum_in{1} = zeros(in, out, 'single');

% Recurrent
net.layers_self = {};
net.regu_self = [];
net.grad_self = {};
net.momentum_self = {};
in = opts.hidden_size;
out = opts.hidden_size;
net.layers_self{1} = 0.01/opts.scale * randn(in, out, 'single');
net.regu_self(1) = single(opts.weightDecay);
net.grad_self{1} = zeros(in, out, 'single');
net.momentum_self{1} = zeros(in, out, 'single');

% Output
net.layers_out = {};
net.regu_out = [];
net.grad_out = {};
net.momentum_out = {};
in = opts.hidden_size;
out = opts.input_size;
net.layers_out{1} = 0.01/opts.scale * randn(in, out, 'single');
net.regu_out(1) = single(opts.weightDecay);
net.grad_out{1} = zeros(in, out, 'single');
net.momentum_out{1} = zeros(in, out, 'single');

% then GPU stuff
if opts.gpu > 0
	% in
	l = length(net.layers_in);
	for i=1:l
		net.layers_in{i} = gpuArray(net.layers_in{i});
		net.grad_in{i} = gpuArray(net.grad_in{i});
		net.momentum_in{i} = gpuArray(net.momentum_in{i});
	end
	net.regu_in = gpuArray(net.regu_in);

	% self
	l = length(net.layers_self);
	for i=1:l
		net.layers_self{i} = gpuArray(net.layers_self{i});
		net.grad_self{i} = gpuArray(net.grad_self{i});
		net.momentum_self{i} = gpuArray(net.momentum_self{i});
	end
	net.regu_self = gpuArray(net.regu_self);

	% out
	l = length(net.layers_out);
	for i=1:l
		net.layers_out{i} = gpuArray(net.layers_out{i});
		net.grad_out{i} = gpuArray(net.grad_out{i});
		net.momentum_out{i} = gpuArray(net.momentum_out{i});
	end
	net.regu_out = gpuArray(net.regu_out);
end

end
