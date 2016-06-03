function [sel] = rnn_gen_album6(net,album,seq,varargin)
    % generation, only support CPU
    opts.num = 1;
    opts.length = 12;
    opts.genK = 500;
    opts.showN = 5;
    opts.head = 10;
    opts.actCut = 50;
    opts = vl_argparse(opts, varargin);
    hidden_size = size(net.layers_self{1},1);
    EOS = [zeros(1,4096) 1];

    seqs = cell(opts.genK,opts.num);
    seqsind = cell(opts.genK,opts.num);
    lls = zeros(opts.genK,opts.num,'single');
    llsall = zeros(opts.genK,opts.num,opts.length,'single');

    maxval = net.opts.max_val;

    hidden_init = cell(1,1);
    hidden_init{1} = 0.5 * ones(1,hidden_size);

    % vocabulary size
    lv = size(net.layers_in{1},1);
    vocab_size=lv;
    input_init = cell(1,1);
    input_init{1} = zeros(1,lv,'single'); %technically not used anyway
    % last one is the End-Of-Sentence (EOS), always start from that
    hidden = hidden_init;
    input = input_init;
    input{1}(:) = EOS;
    alb = [album zeros(size(album,1),1)]/maxval;
    seq = [seq zeros(size(seq,1),1)]/maxval;
    [hidden, output] = rnn_forward(input, hidden, net, opts.actCut);
    for i=1:size(seq,1)
	input{1}(:) = seq(i,:);
	[hidden, output] = rnn_forward(input, hidden, net, opts.actCut);
    end
    finaloutput = rnn_softmax(output{1}*alb'); 
    [~,sel] = max(finaloutput);

