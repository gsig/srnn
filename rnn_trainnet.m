% 
% Script to train the SRNN network  
% The SRNN is implemented on top of a RNN implementation 
% and controls the input and the loss for the RNN network
%
% Gunnar Atli Sigurdsson & Xinlei Chen 2015
% Carnegie Mellon University

% prepare the data
fprintf('Loading data..\n');
datadir = '/nfs/onega_no_backups2/users/gsigurds/storylines_data/';
A = load([datadir,'train_paris.mat'])
A = A.A;
l = length(A);
subsampl = 10+2; %Size of the story. Low number: Easy to learn, but not very informative. High number: hard to learn.
vocab_size = 4096;
input_size = 4096+1;
hidden_size = 50; 
repeatTrain = 10; %This is used to artifically extend the epoch size. Due to randomness this helps make the training more robust.
repeatVal = 10; %Helpful to get a good estimate of the validation likelihood
EOS = [zeros(1,4096) 1];

tmp = cellfun(@(x) max(x(:)),A,'UniformOutput',false);
maxval = max([tmp{:}]);

% start the network
fprintf('Training net..\n');

trainid = 1:round(0.9*l);
validid = round(0.9*l)+1:l;
gpu = 0;

ltr = length(trainid);
lvl = length(validid);

hidden_init = cell(1,1);
hidden_init{1} = 0.5 * ones(1,hidden_size);
ghidden_init = cell(1,1);
ghidden_init{1} = zeros(1,hidden_size,'single');

block = 25; %batch size

Hidden = cell(block,1);
Input = cell(block+1,1); %for the predicted input
Output = cell(block,1);
Outputgrad = cell(block,1);

% learning parameters
iter = 0;
learning_rate = 0.05;
ending_rate = 1e-4;
actCut = 50; %forward pass clip value (clip if <actCut or >actCut
gradCut = 15; %gradient clipping
momentum = 0.9;
last_ll = -inf;
net_final = [];
reduce_thres = 1.003; %reduce LR if new_ll < old_ll*reduce_thres

% select gpu
if gpu > 0
    fprintf('Using GPU ID: %d\n',gpu);
    D = gpuDevice(gpu);
    actCut = gpuArray(actCut);
    gradCut = gpuArray(gradCut);
    momentum = gpuArray(momentum);
    learning_rate = gpuArray(learning_rate);
    hidden_init{1} = gpuArray(hidden_init{1});
    ghidden_init{1} = gpuArray(ghidden_init{1});
else
    D = [];
end

% Initialize the RNN network
net = rnn_initnet('gpu', gpu, 'hidden_size', hidden_size, 'input_size', input_size, 'max_val', maxval, 'repeatTrain', repeatTrain, 'subsampl', subsampl);

while learning_rate > ending_rate
    count = 0;
    ll = 0;
    lstr = 0;

    start = tic;
    rng('shuffle')
    for i=repmat(trainid,1,repeatTrain) % go through training data multiple times before validating because of randomness 
        if size(A{i},1) <= subsampl
            % Skip if small photo album
            continue;
        end

        % initialize the sequence using a special EOS element
        hidden = hidden_init;
        k = 1;
        alb = [EOS; [A{i} zeros(size(A{i},1),1)]; EOS]/maxval;
        ls = size(alb,1);
        output = cell(1,1); output{1} = zeros(1,vocab_size+1); %init not used anyway
        finaloutput = rnn_softmax(ones(1,ls)); %initialize the output space probability
        for j=1:subsampl
	    % select the next sample based on the combined probabilty of the network and the prior
	    % The output space is the future samples
            w = ones(1,size(alb,1));
            w = w.*finaloutput;
            orderprob = ordersample(subsampl,j-1,ls,ls-size(alb,1)+1);
            w = w.*orderprob;
            [~,ind] = sort(w(1:end-1),'descend');
            sprob = w(ind);
            sprob = sprob / sum(sprob);
	    cumprob = cumsum(sprob); cumprob(end)=1;
            if j == 1
                selw = 1;
            elseif j == subsampl
                selw = size(alb,1);
            else
                selw = ind(sum(rand(1) > cumprob) + 1);
            end
            sel = alb(selw,:);

	    % Now that we have selected a sample, we can calculate the loss with respect to that sample
            dzdY = rnn_softmaxlossb(finaloutput,selw,1)*alb;
            alb = alb(selw+1:end,:);
            ll = ll + log2(finaloutput(selw)*length(w)); % the loss is normlized by the length of the output space

	    % Now we move on and calculate a forward pass using the selected next sample
            input = cell(1,1);
            input{1} = sel;
            if gpu > 0
                input{1} = gpuArray(input{1});
                [hidden, output] = rnn_forward(input, hidden, net, actCut);
                finaloutput = rnn_softmax(output{1}*alb'); 
                wait(D);
            else
                [hidden, output] = rnn_forward(input, hidden, net, actCut);
                finaloutput = rnn_softmax(output{1}*alb'); 
            end

            count = count + 1;

            if k == block+1 || j == subsampl
		% If we have have reached the batch size, we backpropagate the gradients 
                Input{k} = input; % add the current input
                Outputgrad{k} = dzdY; % add the current output grad
                ghidden = ghidden_init;
                %this = j + 1;
                % clean up gradient
                net.grad_out{1}(:) = 0;
                net.grad_self{1}(:) = 0;
                net.grad_in{1}(:) = 0;
                n = k-1;
                for k2 = k-1:-1:1
                    gt = Input{k2+1}{1};
                    [ghidden, net] = rnn_backward(Input{k2}, Hidden{k2}, Output{k2}, gt, ghidden, net, gradCut, Outputgrad{k2+1});
                    if gpu > 0
                        wait(D);
                    end
                end
                % update the model parameters using the gradient
                net = rnn_updatenet(net, learning_rate, momentum, n);
                k=1; %reset k
            end

            Input{k} = input;
            Hidden{k} = hidden;
            Output{k} = output;
            Outputgrad{k} = dzdY; 

            k = k + 1;
        end
        % show progress
        if mod(i,10) == 0
            str = sprintf('Iter: %02d Alpha: %.5f Document: %04d LL: %.5f WPS: %.3f',iter,learning_rate,i,ll / count,count / toc(start));
            backSlash = repmat('\b', 1, lstr);
            fprintf(backSlash);
            fprintf(str);
            lstr = length(str);
        end
    end

    % validation, same as before, but no updating of parameters
    count_val = 0;
    ll_val = 0;
    rng(1); % fixing random seed for validatio
    for i=repmat(validid,1,repeatVal)
        if size(A{i},1) <= subsampl
            continue; end
        hidden = hidden_init;
        alb = [EOS; [A{i} zeros(size(A{i},1),1)]; EOS]/maxval;
        ls = size(alb,1);
        output = cell(1,1); output{1} = zeros(1,vocab_size+1); %init not used anyway
        finaloutput = rnn_softmax(ones(1,ls));
        for j=1:subsampl
                % selection
            w = ones(1,size(alb,1));
            w = w.*finaloutput;
            orderprob = ordersample(subsampl,j-1,ls,ls-size(alb,1)+1);
            w = w.*orderprob;
            [~,ind] = sort(w(1:end-1),'descend');
            sprob = w(ind);
            sprob = sprob / sum(sprob);
            cumprob = cumsum(sprob);
            if j == 1
                selw = 1;
            elseif j == subsampl
                selw = size(alb,1);
            else
                selw = ind(sum(rand(1) > cumprob) + 1);
            end
            sel = alb(selw,:);
            dzdY = rnn_softmaxlossb(finaloutput,selw,1)*alb;
            alb = alb(selw+1:end,:);
            ll_val = ll_val + log2(finaloutput(selw)*length(w)); 

            input = cell(1,1);
            input{1} = sel;

            if gpu > 0
                input{1} = gpuArray(input{1});
                [hidden, output] = rnn_forward(input, hidden, net, actCut);
                finaloutput = rnn_softmax(output{1}*alb'); 
                wait(D);
            else
                [hidden, output] = rnn_forward(input, hidden, net, actCut);
                finaloutput = rnn_softmax(output{1}*alb'); 
            end
            count_val = count_val + 1;
        end
    end

    str = sprintf('Iter: %02d Alpha: %.5f Train LL: %.5f Valid LL: %.5f\n',...
        iter,learning_rate,ll / count,ll_val / count_val);
    backSlash = repmat('\b', 1, lstr);
    fprintf(backSlash);
    fprintf(str);

    % If the likelihood is not increasing anymore, we decrease the learning rate
    if ll_val * reduce_thres < last_ll
        learning_rate = learning_rate / 2;
    end

    % Only update the model if the likelihood is better
    if ll_val > last_ll
        last_ll = ll_val;
        net_final = net;
    else
        net = net_final;
    end

    iter = iter + 1;
end

% return to cpu mode
if gpu > 0
    net = rnn_gathernet(net);
    actCut = gather(actCut);
    gradCut = gather(gradCut);
    momentum = gather(momentum);
    learning_rate = gather(learning_rate);
    hidden_init{1} = gather(hidden_init{1});
    ghidden_init{1} = gather(ghidden_init{1});
    ll = gather(ll);
    ll_val = gather(ll_val);

    reset(D);
end


save('tmp.mat','net')
