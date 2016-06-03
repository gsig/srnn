function [ out ] = web_demo( x, concept, folder )
% WEB_DEMO  Code used for an interactive website running SRNN
%           Calculates fc7 features for images using CAFFE
%           And generates a summary using trained SRNN models
%           The runtime of the demo on CPU is around 20 seconds 
%           if CAFFE is installed with MKL enabled
%   
% Gunnar Atli Sigurdsson & Xinlei Chen 2015
% Carnegie Mellon University

cachedir = '/home/gsigurds/cachedir';
fid = fopen([folder filesep 'album.txt']);
C = textscan(fid,'%s'); C=C{1};
%N = min(100,length(C)); % speedup for large albums
%subset = round(linspace(1,length(C),N));
%C = C(subset);
pop = @(x) x{end};
foldername = pop(strsplit(folder,'/'));
cachepath = [cachedir filesep foldername '.mat'];
Clocal = cellfun(@(x) [folder filesep pop(strsplit(x,'/'))], C,'UniformOutput',false);
Cname = cellfun(@(x) pop(strsplit(x,'/')), C,'UniformOutput',false);
tic;
if exist(cachepath, 'file')==2
    A = load(cachepath); A = A.A;
elseif x==1
    A = extract_feat(Clocal);
    save(cachepath,'A');
else
    while(exist(cachepath, 'file')~=2)
        pause(.5);
    end
    for i=1:5
        try 
            A = load(cachepath); A = A.A;
        catch exception
            pause(1);
        end
    end
end
toc;
if x==1
    ind = 1;
else
    rng(123);
    ind = summarize(x,concept,A);
end
out = strjoin(Cname(ind),'\n');
fprintf(out);
fprintf('\n');

end

function [A] = extract_feat( files )
    BATCH_SIZE = 150; 
    CROPPED_DIM = 227;
    N = length(files);
    caffedir = '/home/gsigurds/caffe';
    addpath([caffedir filesep 'matlab']);
    caffe.set_mode_cpu();
    model_dir = [caffedir '/models/bvlc_reference_caffenet/'];
    net_model = [model_dir 'deploy.prototxt'];
    net_weights = [model_dir 'bvlc_reference_caffenet.caffemodel'];
    phase = 'test';
    net = caffe.Net(net_model, net_weights, phase);
    d = load([caffedir filesep '/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat']);
    mean_data = d.mean_data;
    net.blobs('data').reshape([227 227 3 BATCH_SIZE]); % reshape blob 'data'
    net.reshape();

    A = zeros(N,4096);
    batches = cell(ceil(N/BATCH_SIZE),1);
    for i=1:length(batches)
        batches{i} = (i-1)*100+1:min(i*100,N);
    end
    input_data = zeros(CROPPED_DIM, CROPPED_DIM, 3, BATCH_SIZE, 'single');
    for b=1:length(batches)
        batch = batches{b};
        for ii=1:length(batch)
            i = batch(ii);
            im = imread(files{i}); 
            tmp = prepare_image(im,mean_data);
            input_data(:,:,:,ii) = tmp;
        end
        net.forward({input_data});
        fc7 = net.blobs('fc7').get_data();
        A(batch,:) = fc7(:,1:length(batch))';
    end
        
end


function crops_data = prepare_image(im,mean_data)
% caffe/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat contains mean_data that
% is already in W x H x C with BGR channels
%d = load('../+caffe/imagenet/ilsvrc_2012_mean.mat');
%mean_data = d.mean_data;
IMAGE_DIM = 256;
CROPPED_DIM = 227;

% Convert an image returned by Matlab's imread to im_data in caffe's data
% format: W x H x C with BGR channels
im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
im_data = permute(im_data, [2, 1, 3]);  % flip width and height
im_data = single(im_data);  % convert from uint8 to single
im_data = imresize(im_data, [IMAGE_DIM IMAGE_DIM], 'bilinear');  % resize im_data
im_data = im_data - mean_data;  % subtract mean_data (already in W x H x C, BGR)

% oversample (4 corners, center, and their x-axis flips)
indices = [0 IMAGE_DIM-CROPPED_DIM] + 1;
crops_data = zeros(CROPPED_DIM, CROPPED_DIM, 3, 1, 'single');
center = floor(indices(2) / 2) + 1;
crops_data(:,:,:,1) = im_data(center:center+CROPPED_DIM-1,center:center+CROPPED_DIM-1,:);

end


function [ind] = summarize(summary,concept,A)
    olddir = pwd;
    CODEDIR = '.';
    cd(CODEDIR);
    if strcmp(concept,'paris')
        if summary == 5
        load('modelparis5.mat');
        elseif summary == 10
        load('testorig5rep.mat');
        elseif summary == 20
        load('modelparis20.mat');
        end
    elseif strcmp(concept,'wedding')
        if summary == 5
        load('modelwedding5.mat');
        elseif summary == 10
        load('originalwedding1.mat');
        elseif summary == 20
        load('modelwedding20.mat');
        end
    elseif strcmp(concept,'london')
        if summary == 5
        load('modellondon5.mat');
        elseif summary == 10
        load('originallondon2.mat');
        elseif summary == 20
        load('modellondon20.mat');
        end
    elseif strcmp(concept,'christmas')
        if summary == 5
        load('modelchristmas5.mat');
        elseif summary == 10
        load('orignalchristmas.mat');
        elseif summary == 20
        load('modelchristmas20.mat');
        end
    elseif strcmp(concept,'safari')
        if summary == 5
        load('modelsafari5.mat');
        elseif summary == 10
        load('modelsafari.mat');
        elseif summary == 20
        load('modelsafari20.mat');
        end
    elseif strcmp(concept,'scuba')
        if summary == 5
        load('modelscuba5.mat');
        elseif summary == 10
        load('modelscuba.mat');
        elseif summary == 20
        load('modelscuba20.mat');
        end
    elseif summary == 7
        load('modelfuneral.mat');
    elseif strcmp(concept,'snowboarding')
        load('modelsnowboarding.mat');
    elseif summary == 9
        load('modelwimbledon.mat');
    elseif summary == 10
        load('modelrock.mat');
    end

    [outseq77,outll77,outseq77ind,outllall77] = rnn_gen_album(net,A,'length',summary+2,'genK',500);
    best = outseq77ind{1};
    ind = best(2:end-1);
    cd(olddir);
end
 
