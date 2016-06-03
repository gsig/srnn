%
% Generate summaries for multiple photo albums using the trained SRNN
% Calls RNN_GEN_ALBUM to generate a likely sequence using the SRNN which is the summary
%
% Gunnar Atli Sigurdsson & Xinlei Chen 2015
% Carnegie Mellon University

datadir = '/nfs/onega_no_backups2/users/gsigurds/storylines_data/';
outputname = 'output.mat';
outputnamebest = 'outputbest.mat';
summaries = cell(50);
scores = nan(50,1);
clear A
load('paris10model.mat');

    
%if ~(matlabpool('size') > 0)
%    matlabpool(6); 
%end
%parfor i = 1:length(A)
for i = 1:length(A)
    fprintf('%d\n',i);
    example = A{i};
    if size(example,1) < 20; continue; end;
    urls = S{i};
    [outseq,outll,outseqind,outllall] = rnn_gen_album(net,example,'length',10+2,'genK',1000);
    best = outseqind{1};
    bestscore = outll(1);
    assert(best(end)-1 == length(urls));
    indices = best(2:end-1)-1; %0 offset
    summaries{i} = urls(best(2:end-1)+1-1); %inds should be one offset for urls, then back for [EOS album EOS]
    scores(i) = bestscore;
end

save(outputname,'summaries');
[~,i] = nanmax(scores);
fprintf('best score: %d', i);
bestsummary = summaries{i};
save(outputnamebest,'bestsummary'),

