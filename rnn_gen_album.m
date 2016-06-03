function [seqs,lls,seqsind,llsall] = rnn_gen_album(net,album,varargin)
% RNN_GEN_ALBUM  Samples multiple times from the RNN using the given data and reports likeliest one. This generates a summary
%   [seqs,lls,seqsind,llsall] = rnn_gen_album(net,album,varargin)
%   
%   album: Nx4096 matrix of fc7 features for images in the album
%   
% Gunnar Atli Sigurdsson & Xinlei Chen 2015
% Carnegie Mellon University

    % generation, only support CPU
    opts.num = 1; %number of runs
    opts.length = 10+2;
    opts.genK = 1000; %number of samples in each run
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
    % last one is the End-Of-Sentence (EOS), always start from that
    input_init = cell(1,1);
    input_init{1} = zeros(1,lv,'single');
    input_init{1}(:) = EOS; 
    alpha = zeros(1,lv);
    alb = [EOS; [album zeros(size(album,1),1)]; EOS]/maxval; % add end of sentence to start/end
    for r=1:opts.num %multiple runs
        for i=1:opts.genK %multiple samples
        % set up network
            hidden = hidden_init;
            input = input_init;
            modelcount = 0;
            seqs{i,r} = zeros(1,opts.length,'int32');
            seqsind{i,r} = zeros(1,opts.length,'int32');
            seqs{i,r}(1) = 0; 
            seqsind{i,r}(1) = 0; 
            selw = 1; %init index 
            output = cell(1,1); output{1} = zeros(1,vocab_size+1);
            for j=2:opts.length 
                % generate the next sample in the sequence
                [hidden, output] = rnn_forward(input, hidden, net, opts.actCut);

                % using the distribution of the output space, select the next sample
                tmp = alb*output{1}'; tmp = tmp(selw+1:end);
                finaloutput = exp(tmp-max(tmp)); finaloutput = finaloutput'./sum(finaloutput);
                oldselw = selw;
                w = ones(1,size(alb,1));
                w(selw+1:end) = w(selw+1:end).*finaloutput;
                w(1:selw) = 0;
                orderprob = fastiterativeOrdSubProb(opts.length,j-1,size(alb,1),selw+1); %probably of ordered subset
                w = w.*orderprob;
                sprob = w / sum(w);
                cumprob = cumsum(sprob); cumprob(end) = 1+eps;
                if j == 1
                    selw = 1;
                elseif j == opts.length
                    selw = size(alb,1);
                else
                    selw = (sum(rand(1) > cumprob))+1;
                end
                sel = alb(selw,:);

                % store results and calculate likelihood
                input{1}(:) = sel;
                seqs{i,r}(j) = 0;
                seqsind{i,r}(j) = selw-1; %adjusting for one offset 
                lls(i,r) = lls(i,r) + log2(finaloutput(selw-oldselw)*length(finaloutput));
                llsall(i,r,j) = log2(finaloutput(selw-oldselw)*length(finaloutput));
            end
        end
    end

    % top ones
    [lls,inds] = sort(lls,1,'descend');
    for r=1:opts.num
        seqs(:,r) = seqs(inds(:,r),r);
        seqsind(:,r) = seqsind(inds(:,r),r);
        llsall(:,r,:) = llsall(inds(:,r),r,:);
    end
    seqs = seqs(1:opts.showN,:);
    seqsind = seqsind(1:opts.showN,:);
    lls = lls(1:opts.showN,:);
    llsall = llsall(1:opts.showN,:,:);

end

function [prob] = iterativeOrdSubProb(K,numbersPicked,N,current)
    prob = zeros(1,N);
    not = 1;
    for i=current:N-(K-numbersPicked)+1
        s = (K-numbersPicked)/(N-i+1);
    prob(i) = s*not;
    not = not * (1-s);
    end
    %prob = prob(current:N);
end

function [prob] = fastiterativeOrdSubProb(K,numbersPicked,N,current)
    prob = zeros(1,N);
    ind = current:N-(K-numbersPicked)+1;
    ss = (K-numbersPicked)./(N-ind+1);
    nots = [1 cumprod(1-ss(1:end-1))];
    prob(ind) = ss.*nots;
    %prob = prob(current:N);
end

