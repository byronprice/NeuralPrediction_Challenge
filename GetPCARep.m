function [] = GetPCARep()
warning('off','all');

cellinfo

numCells = length(celldata);

neuralResponse = cell(numCells,1);
for ii=1:numCells
    load(celldata(ii).datafile,'resp');
    neuralResponse{ii} = resp;
end


for ii=1:numCells
    resp = neuralResponse{ii};
    
    inds = find(~isnan(resp));
    newResp = resp(inds);
    numFrames = length(newResp);
    mov = loadimfile(celldata(ii).fullstimfile);
    
    [DIM,~,fullFrames] = size(mov);
    meanIm = mean(mov,3);
%     for jj=1:fullFrames
%         mov(:,:,jj) = mov(:,:,jj) - meanIm;
%     end
    
    valMov = loadimfile(celldata(ii).fullvalstimfile);
    [DIM1,~,valFrames] = size(valMov);
    
    newValMov = zeros(DIM,DIM,valFrames);
    for jj=1:valFrames
        temp = valMov(:,:,jj);
        if DIM1 ~= DIM
           temp = imresize(temp,DIM/DIM1); 
        end
%         temp = temp-meanIm;
        newValMov(:,:,jj) = temp;
    end
    valMov = newValMov;clear newValMov DIM1;
    cellid = celldata(ii).cellid;
    numBack = 20;
    [pcaTrain,pcaVal,Winv,Q,mu] = DCTModel(mov,inds,DIM,numBack,numFrames,valFrames,valMov,meanIm);
    save(sprintf('Cell%d_DCTPCA.mat',ii),'Winv','Q','mu',...
        'inds','resp','newresp','pcaTrain','pcaVal','cellid','DIM','numBack');
    
    [pcaTrain,pcaVal,Winv,Q,mu] = WVLTModel(mov,inds,DIM,numBack,numFrames,valFrames,valMov,meanIm);
    save(sprintf('Cell%d_WVLTPCA.mat',ii),'Winv','Q','mu',...
        'inds','resp','newresp','pcaTrain','pcaVal','cellid','DIM','numBack');
    
    params.tsize = numBack;
    params.phasemode = 3;
    params.phasemode_sfmax = Inf;
    params.fdivisions = 10;
    params.veldivisions = 10;
    params.sfmax = 10;
    params.sfmin = 1;
    params.tfmax = 5;
    params.dirdivisions = 12;
    
    newMov = zeros(DIM,DIM,fullFrames+valFrames+numBack-1);
    
    count = 1;
    for jj=1:fullFrames
        newMov(:,:,count) = mov(:,:,jj);
        count = count+1;
    end
    for jj=1:numBack-1
       newMov(:,:,count) = meanIm;
       count = count+1;
    end
    for jj=1:valFrames
        newMov(:,:,count) = valMov(:,:,jj);
        count = count+1;
    end
        
    [stim,params] = preprocWavelets3d(newMov,params);
    trainStim = stim(1:fullFrames,:);
    valStim = stim(fullFrames+numBack:end,:);
    save(sprintf('Cell%d_3Dwvlts20.mat',ii),'inds','resp','newresp','trainStim',...
        'params','valStim','cellid','DIM','numBack');
    
    numBack = 10;
    params.tsize = 10;
    [stim,params] = preprocWavelets3d(newMov,params);
    trainStim = stim(1:fullFrames,:);
    valStim = stim(fullFrames+numBack:end,:);
    save(sprintf('Cell%d_3Dwvlts10.mat',ii),'inds','resp','newresp','trainStim',...
        'params','valStim','cellid','DIM','numBack');
end
end

function [pcaTrain,pcaVal,Winv,Q,mu] = DCTModel(mov,inds,DIM,numBack,numFrames,valFrames,valMov,meanIm)
% DCT-PCA model
dimRun = [100,90,80,70,60,50,40,30,20,10.*ones(1,numBack-7)];
%dimRun = 30.*ones(1,numBack);
dctDims = zeros(DIM,DIM,numBack);
for jj=1:numBack
    currentDim = min(DIM,dimRun(jj));
    temp = ones(currentDim,currentDim);
    dctDims(1:currentDim,1:currentDim,jj) = temp;
end
fullDctDim = sum(dctDims(:));dctDims = find(dctDims);
dctMov = zeros(numFrames+valFrames,fullDctDim);

count = 1;
for jj=inds'
    miniMov = zeros(DIM,DIM,numBack);
    for kk=1:numBack
        if jj-kk-1 < 1
            temp = meanIm;
        else
            temp = mov(:,:,max(jj-kk-1,1));
        end
        miniMov(:,:,kk) = temp;
    end
    R = mirt_dctn(miniMov);
    R = R(dctDims);
    dctMov(count,:) = R(:);
    count = count+1;
end

for jj=1:valFrames
    miniMov = zeros(DIM,DIM,numBack);
    for kk=1:numBack
        if jj-kk-1 < 1
            temp = meanIm;
        else
            temp = valMov(:,:,max(jj-kk-1,1));
        end
        miniMov(:,:,kk) = temp;
    end
    R = mirt_dctn(miniMov);
    R = R(dctDims);
    dctMov(count,:) = R(:);
    count = count+1;
end

S = cov(dctMov); % or try shrinkage_cov
% S = shrinkage_cov(dctMov);
[V,D] = eig(S);clear S;

dctMov = dctMov';
mu = mean(dctMov,2);
dctMov = dctMov-repmat(mu,[1,numFrames+valFrames]);

allEigs = diag(D);
fullVariance = sum(allEigs);
for jj=50:1:fullDctDim-10
    start = fullDctDim-jj+1;
    eigenvals = allEigs(start:end);
    varianceProp = sum(eigenvals)/fullVariance;
    if varianceProp >= 0.99
        break;
    end
end

Q = jj;
meanEig = mean(allEigs(1:start-1));
W = V(:,start:end)*sqrtm(D(start:end,start:end)-meanEig.*eye(Q));
clear V D;
W = fliplr(W);
Winv = pinv(W);
x = Winv*dctMov; % number of dimensions kept by N
reduceDctData = x';

pcaTrain = reduceDctData(1:numFrames,:);
pcaVal = reduceDctData(numFrames+1:end,:);

end

function [pcaTrain,pcaVal,Winv,Q,mu] = WVLTModel(mov,inds,DIM,numBack,numFrames,valFrames,valMov,meanIm)
% wavelet-PCA model
miniMov = zeros(DIM,DIM,numBack);
R = wavedec3(miniMov,2,'db4');
R = R.dec;

fullWVLTdim = 0;

for ii=1:size(R,1)
    fullWVLTdim = fullWVLTdim+numel(R{ii});
end

wvltMov = zeros(numFrames+valFrames,fullWVLTdim);

count = 1;
for jj=inds'
    miniMov = zeros(DIM,DIM,numBack);
    for kk=1:numBack
        if jj-kk-1 < 1
            temp = meanIm;
        else
            temp = mov(:,:,max(jj-kk-1,1));
        end
        miniMov(:,:,kk) = temp;
    end
    R = wavedec3(miniMov,2,'db4');
    R = R.dec;
    
    new = [];
    for kk=1:size(R,1)
        temp = R{kk};
        new = [new;temp(:)];
    end
    wvltMov(count,:) = new;
    count = count+1;
end

for jj=1:valFrames
    miniMov = zeros(DIM,DIM,numBack);
    for kk=1:numBack
        if jj-kk-1 < 1
            temp = meanIm;
        else
            temp = valMov(:,:,max(jj-kk-1,1));
        end
        miniMov(:,:,kk) = temp;
    end
    R = wavedec3(miniMov,2,'db4');
    R = R.dec;
    
    new = [];
    for kk=1:size(R,1)
        temp = R{kk};
        new = [new;temp(:)];
    end
    wvltMov(count,:) = new;
    count = count+1;
end

%wvltMov = wvltMov-mean(wvltMov,1);

maxToKeep = 4e4;p = maxToKeep/fullWVLTdim;
temp = var(wvltMov,[],1);
Q = quantile(temp,1-p);
tempInds = temp>=Q;
wvltMov = wvltMov(:,tempInds);
fullWVLTdim = size(wvltMov,2);
S = cov(wvltMov); % or try shrinkage_cov
% S = shrinkage_cov(dctMov);
[V,D] = eig(S);clear S;

wvltMov = wvltMov';
mu = mean(wvltMov,2);
wvltMov = wvltMov-repmat(mu,[1,numFrames+valFrames]);

allEigs = diag(D);
fullVariance = sum(allEigs);
for jj=50:1:fullWVLTdim-10
    start = fullWVLTdim-jj+1;
    eigenvals = allEigs(start:end);
    varianceProp = sum(eigenvals)/fullVariance;
    if varianceProp >= 0.99
        break;
    end
end

Q = jj;
meanEig = mean(allEigs(1:start-1));
W = V(:,start:end)*sqrtm(D(start:end,start:end)-meanEig.*eye(Q));
clear V D;
W = fliplr(W);
Winv = pinv(W);
x = Winv*wvltMov; % number of dimensions kept by N
reduceWvltData = x';

pcaTrain = reduceWvltData(1:numFrames,:);
pcaVal = reduceWvltData(numFrames+1:end,:);

end