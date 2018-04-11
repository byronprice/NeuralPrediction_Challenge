function [] = TestRF()
warning('off','all');

train = 0.7;

cellinfo

numCells = length(celldata);

neuralResponse = cell(numCells,1);
for ii=1:numCells
    load(celldata(ii).datafile,'resp');
    neuralResponse{ii} = resp;
end


for ii=1:numCells
    resp = neuralResponse{ii};
    
    histLags = 30;
    histDesign = zeros(length(resp),histLags);
    temp = resp;
    for jj=1:histLags
        temp = [0;temp(1:end-1)];
        histDesign(:,jj) = temp;
    end
    
    inds = find(~isnan(sum(histDesign,2)) & ~isnan(resp));
    newResp = resp(inds);
    histDesign = histDesign(inds,:);
    numFrames = length(newResp);
    mov = loadimfile(celldata(ii).fullstimfile);
    
    [DIM,~,fullFrames] = size(mov);
    meanIm = mean(mov,3);
    for jj=1:fullFrames
        mov(:,:,jj) = mov(:,:,jj) - meanIm;
    end
    
    numBack = 30;
    
    histBases = 10;
    histdev = 2;
    
    histBasisFuns = zeros(histLags,histBases);
    time = linspace(0,histLags-1,histLags);
    centerPoints = linspace(0,histLags-1,histBases);
    for kk=1:histBases
        histBasisFuns(:,kk) = exp(-(time-centerPoints(kk)).^2./(2*histdev*histdev));
    end
    
    histDesign = histDesign*histBasisFuns;
     
    [predictTrain,trueTrain,expDev,b,Q,mu,Winv] = DCTModel(mov,inds,DIM,numBack,numFrames,newResp,histDesign,train);
    save(sprintf('Cell%d_DCT.mat',ii),'predictTrain','trueTrain','expDev',...
        'b','Q','histDesign','histBasisFuns','mu','Winv');
    fprintf('Cell: %d - DCT Explain Dev: %3.1f\n',ii,expDev);
    
    [predictTrain,trueTrain,expDev,b,Q,mu,Winv] = WVLTModel(mov,inds,DIM,numBack,numFrames,newResp,histDesign,train);
    save(sprintf('Cell%d_WVLT.mat',ii),'predictTrain','trueTrain','expDev',...
        'b','Q','histDesign','histBasisFuns','mu','Winv');
    fprintf('Cell: %d - WVLT Explain Dev: %3.1f\n',ii,expDev);
    
    [predictTrain,trueTrain,expDev,b,Q,mu,Winv] = DCTOASModel(mov,inds,DIM,numBack,numFrames,newResp,histDesign,train);
    save(sprintf('Cell%d_DCTOAS.mat',ii),'predictTrain','trueTrain','expDev',...
        'b','Q','histDesign','histBasisFuns','mu','Winv');
    fprintf('Cell: %d - DCT-OAS Explain Dev: %3.1f\n',ii,expDev);
    
    [predictTrain,trueTrain,expDev,b,Q,mu,Winv] = WVLTOASModel(mov,inds,DIM,numBack,numFrames,newResp,histDesign,train);
    save(sprintf('Cell%d_WVLTOAS.mat',ii),'predictTrain','trueTrain','expDev',...
        'b','Q','histDesign','histBasisFuns','mu','Winv');
    fprintf('Cell: %d - WVLT-OAS Explain Dev: %3.1f\n',ii,expDev);
end
end

function [predictTrain,trueTrain,expDev,b,Q,mu,Winv] = DCTModel(mov,inds,DIM,numBack,numFrames,newResp,histDesign,train)
% DCT-PCA model
dimRun = [90,80,70,60,50,40,30,20,10.*ones(1,numBack-8)];
dctDims = zeros(DIM,DIM,numBack);
for jj=1:numBack
    currentDim = min(DIM,dimRun(jj));
    temp = ones(currentDim,currentDim);
    dctDims(1:currentDim,1:currentDim,jj) = temp;
end
fullDctDim = sum(dctDims(:));dctDims = find(dctDims);
dctMov = zeros(numFrames,fullDctDim);

count = 1;
for jj=inds'
    miniMov = zeros(DIM,DIM,numBack);
    for kk=1:numBack
        if jj-kk-2 < 1
            temp = zeros(DIM,DIM);
        else
            temp = mov(:,:,max(jj-kk-2,1));
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
dctMov = dctMov-repmat(mu,[1,numFrames]);

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

trainN = round(length(newResp).*train);

testPCAdims = unique(round(linspace(0,Q,20)));
AIC = zeros(length(testPCAdims),1);
allB = cell(length(testPCAdims),1);

count = 1;
for ii=testPCAdims
    [b,dev,~] = glmfit([histDesign(1:trainN,:),...
            reduceDctData(1:trainN,1:ii)],newResp(1:trainN),'poisson');
    AIC(count) = dev+2*length(b);
    allB{count} = b;
    count = count+1;
end

[~,ind] = min(AIC);

b = allB{ind};Q = testPCAdims(ind);
numHistParams = size(histDesign,2);

predictTrain = exp(b(1)+reduceDctData(trainN+1:end,1:Q)*b(2+numHistParams:end));
trueTrain = newResp(trainN+1:end);

initialDev = trueTrain.*log(trueTrain./predictTrain)-(trueTrain-predictTrain);
initialDev(isnan(initialDev) | isinf(initialDev)) = predictTrain(isnan(initialDev) | isinf(initialDev));
modelDev = 2*sum(initialDev);

nullEst = mean(trueTrain).*ones(length(trueTrain),1);
nullDev = trueTrain.*log(trueTrain./nullEst)-(trueTrain-nullEst);
nullDev(isnan(nullDev) | isinf(nullDev)) = nullEst(isnan(nullDev) | isinf(nullDev));
nullDev = 2*sum(nullDev);

expDev = max(1-modelDev/nullDev,0);
end


function [predictTrain,trueTrain,expDev,b,Q,mu,Winv] = WVLTModel(mov,inds,DIM,numBack,numFrames,newResp,histDesign,train)
% wavelet-PCA model
miniMov = zeros(DIM,DIM,numBack);
R = wavedec3(miniMov,1,'db4');
R = R.dec;R = R{1};

fullWVLTDim = numel(R)*4;
wvltMov = zeros(numFrames,fullWVLTDim);

count = 1;
for jj=inds'
    miniMov = zeros(DIM,DIM,numBack);
    for kk=1:numBack
        if jj-kk-2 < 1
            temp = zeros(DIM,DIM);
        else
            temp = mov(:,:,max(jj-kk-2,1));
        end
        miniMov(:,:,kk) = temp;
    end
    R = wavedec3(miniMov,1,'db4');
    R = R.dec;
    
    new = [];
    for kk=[1,2,3,5]
        temp = R{kk};
        new = [new;temp(:)];
    end
    wvltMov(count,:) = new;
    count = count+1;
end

tempInds = var(wvltMov,[],1)>100;
wvltMov = wvltMov(:,tempInds);
S = cov(wvltMov); % or try shrinkage_cov
% S = shrinkage_cov(dctMov);
[V,D] = eig(S);clear S;

wvltMov = wvltMov';
mu = mean(wvltMov,2);
wvltMov = wvltMov-repmat(mu,[1,numFrames]);

allEigs = diag(D);
fullVariance = sum(allEigs);
for jj=50:1:fullWVLTDim-10
    start = fullWVLTDim-jj+1;
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
reduceDctData = x';

trainN = round(length(newResp).*train);

testPCAdims = unique(round(linspace(0,Q,20)));
AIC = zeros(length(testPCAdims),1);
allB = cell(length(testPCAdims),1);

count = 1;
for ii=testPCAdims
    [b,dev,~] = glmfit([histDesign(1:trainN,:),...
            reduceDctData(1:trainN,1:ii)],newResp(1:trainN),'poisson');
    AIC(count) = dev+2*length(b);
    allB{count} = b;
    count = count+1;
end

[~,ind] = min(AIC);

b = allB{ind};Q = testPCAdims(ind);
numHistParams = size(histDesign,2);

predictTrain = exp(b(1)+reduceDctData(trainN+1:end,1:Q)*b(2+numHistParams:end));
trueTrain = newResp(trainN+1:end);

initialDev = trueTrain.*log(trueTrain./predictTrain)-(trueTrain-predictTrain);
initialDev(isnan(initialDev) | isinf(initialDev)) = predictTrain(isnan(initialDev) | isinf(initialDev));
modelDev = 2*sum(initialDev);

nullEst = mean(trueTrain).*ones(length(trueTrain),1);
nullDev = trueTrain.*log(trueTrain./nullEst)-(trueTrain-nullEst);
nullDev(isnan(nullDev) | isinf(nullDev)) = nullEst(isnan(nullDev) | isinf(nullDev));
nullDev = 2*sum(nullDev);

expDev = max(1-modelDev/nullDev,0);

end

function [predictTrain,trueTrain,expDev,b,Q,mu,Winv] = DCTOASModel(mov,inds,DIM,numBack,numFrames,newResp,histDesign,train)
% DCT-PCA model
dimRun = [90,80,70,60,50,40,30,20,10.*ones(1,numBack-8)];
dctDims = zeros(DIM,DIM,numBack);
for jj=1:numBack
    currentDim = min(DIM,dimRun(jj));
    temp = ones(currentDim,currentDim);
    dctDims(1:currentDim,1:currentDim,jj) = temp;
end
fullDctDim = sum(dctDims(:));dctDims = find(dctDims);
dctMov = zeros(numFrames,fullDctDim);

count = 1;
for jj=inds'
    miniMov = zeros(DIM,DIM,numBack);
    for kk=1:numBack
        if jj-kk-2 < 1
            temp = zeros(DIM,DIM);
        else
            temp = mov(:,:,max(jj-kk-2,1));
        end
        miniMov(:,:,kk) = temp;
    end
    R = mirt_dctn(miniMov);
    R = R(dctDims);
    dctMov(count,:) = R(:);
    count = count+1;
end

% S = cov(dctMov); % or try shrinkage_cov
[S,~] = shrinkage_cov(dctMov);
[V,D] = eig(S);clear S;

dctMov = dctMov';
mu = mean(dctMov,2);
dctMov = dctMov-repmat(mu,[1,numFrames]);

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

trainN = round(length(newResp).*train);

testPCAdims = unique(round(linspace(0,Q,20)));
AIC = zeros(length(testPCAdims),1);
allB = cell(length(testPCAdims),1);

count = 1;
for ii=testPCAdims
    [b,dev,~] = glmfit([histDesign(1:trainN,:),...
            reduceDctData(1:trainN,1:ii)],newResp(1:trainN),'poisson');
    AIC(count) = dev+2*length(b);
    allB{count} = b;
    count = count+1;
end

[~,ind] = min(AIC);

b = allB{ind};Q = testPCAdims(ind);
numHistParams = size(histDesign,2);

predictTrain = exp(b(1)+reduceDctData(trainN+1:end,1:Q)*b(2+numHistParams:end));
trueTrain = newResp(trainN+1:end);

initialDev = trueTrain.*log(trueTrain./predictTrain)-(trueTrain-predictTrain);
initialDev(isnan(initialDev) | isinf(initialDev)) = predictTrain(isnan(initialDev) | isinf(initialDev));
modelDev = 2*sum(initialDev);

nullEst = mean(trueTrain).*ones(length(trueTrain),1);
nullDev = trueTrain.*log(trueTrain./nullEst)-(trueTrain-nullEst);
nullDev(isnan(nullDev) | isinf(nullDev)) = nullEst(isnan(nullDev) | isinf(nullDev));
nullDev = 2*sum(nullDev);

expDev = max(1-modelDev/nullDev,0);
end

function [predictTrain,trueTrain,expDev,b,Q,mu,Winv] = WVLTOASModel(mov,inds,DIM,numBack,numFrames,newResp,histDesign,train)
% wavelet-PCA model
miniMov = zeros(DIM,DIM,numBack);
R = wavedec3(miniMov,1,'db4');
R = R.dec;R = R{1};

fullWVLTDim = numel(R)*4;
wvltMov = zeros(numFrames,fullWVLTDim);

count = 1;
for jj=inds'
    miniMov = zeros(DIM,DIM,numBack);
    for kk=1:numBack
        if jj-kk-2 < 1
            temp = zeros(DIM,DIM);
        else
            temp = mov(:,:,max(jj-kk-2,1));
        end
        miniMov(:,:,kk) = temp;
    end
    R = wavedec3(miniMov,1,'db4');
    R = R.dec;
    
    new = [];
    for kk=[1,2,3,5]
        temp = R{kk};
        new = [new;temp(:)];
    end
    wvltMov(count,:) = new;
    count = count+1;
end

tempInds = var(wvltMov,[],1)>100;
wvltMov = wvltMov(:,tempInds);
% S = cov(wvltMov); % or try shrinkage_cov
S = shrinkage_cov(wvltMov);
[V,D] = eig(S);clear S;

wvltMov = wvltMov';
mu = mean(wvltMov,2);
wvltMov = wvltMov-repmat(mu,[1,numFrames]);

allEigs = diag(D);
fullVariance = sum(allEigs);
for jj=50:1:fullWVLTDim-10
    start = fullWVLTDim-jj+1;
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
reduceDctData = x';

trainN = round(length(newResp).*train);

testPCAdims = unique(round(linspace(0,Q,20)));
AIC = zeros(length(testPCAdims),1);
allB = cell(length(testPCAdims),1);

count = 1;
for ii=testPCAdims
    [b,dev,~] = glmfit([histDesign(1:trainN,:),...
            reduceDctData(1:trainN,1:ii)],newResp(1:trainN),'poisson');
    AIC(count) = dev+2*length(b);
    allB{count} = b;
    count = count+1;
end

[~,ind] = min(AIC);

b = allB{ind};Q = testPCAdims(ind);
numHistParams = size(histDesign,2);

predictTrain = exp(b(1)+reduceDctData(trainN+1:end,1:Q)*b(2+numHistParams:end));
trueTrain = newResp(trainN+1:end);

initialDev = trueTrain.*log(trueTrain./predictTrain)-(trueTrain-predictTrain);
initialDev(isnan(initialDev) | isinf(initialDev)) = predictTrain(isnan(initialDev) | isinf(initialDev));
modelDev = 2*sum(initialDev);

nullEst = mean(trueTrain).*ones(length(trueTrain),1);
nullDev = trueTrain.*log(trueTrain./nullEst)-(trueTrain-nullEst);
nullDev(isnan(nullDev) | isinf(nullDev)) = nullEst(isnan(nullDev) | isinf(nullDev));
nullDev = 2*sum(nullDev);

expDev = max(1-modelDev/nullDev,0);

end

function [sigma,rho] = shrinkage_cov(X,est)
% this program is distributed under BSD 2-Clause license
%
% Copyright (c) <2016>, <Okba BEKHELIFI, okba.bekhelifi@univ-usto.dz>
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
%
% 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
%
% 2. Redistributions in binary form must reproduce the above copyright notice,
%    this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
%
%
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
% BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
% IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
% OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
% OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
% OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%
%
%compute a covariance matrix estimation using shrinkage estimators: RBLW or
% OAS describer in:
%     [1] "Shrinkage Algorithms for MMSE Covariance Estimation"
%     Chen et al., IEEE Trans. on Sign. Proc., Volume 58, Issue 10, October 2010.
%Input:
%X: data NxP : N : samples P: features
%est: shrinkage estimator
%     'rblw' : Rao-Blackwell estimator
%     'oas'  : oracle approximating shrinkage estimator
%     default estimator is OAS.
%
%Output:
% sigma: estimated covariance matrix
% rho : shrinkage coefficient
%
%created: 02/05/2016
%last revised: 14/06/2016


[n,p] = size(X);
% sample covariance, formula (2) in the paper [1]
X = bsxfun(@minus,X,mean(X));
S = X'*X/(n-1);
% structured estimator, formula (3) in the article [1]
mu = trace(S)/p;
F = mu*eye(p);


if (nargin < 2) est = 'oas';
end
switch lower(est)
    
    case 'oas'
        
%         rho = (1-(2/p)*trace(S^2)+trace(S)^2)/((n+1-2/p)*(trace(S^2)-1/p*trace(S)^2));
        c1 = 1-2/p;
        c2 = n+1-2/p;
        c3 = 1-n/p;
        rho = (c1*trace(S^2) + trace(S)^2) / (c2*trace(S^2) + c3*trace(S)^2);
        
    case 'rblw'
        
        c1 = (n-2)/n;
        c2 = n+2;
        rho = ( c1*trace(S^2)+trace(S)^2 )/( c2*( trace(S^2)-(trace(S)^2/p) ) );
        
    otherwise 'Shrinkage estimator not provided correctly';
        
end

% regularization, formula (4) in the paper [1]
sigma = (1-rho)*S + rho*F;

end