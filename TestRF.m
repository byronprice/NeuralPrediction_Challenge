function [] = TestRF()
% glm encoding models after doing DCT or wavelet decomposition of videos
% and applying PCA to reduce dimensionality
warning('off','all');

train = 0.8;

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
    
    % validation set
    valMov = loadimfile(celldata(ii).fullvalstimfile);
    [DIM1,~,valFrames] = size(valMov);
    newValMov = zeros(DIM,DIM,valFrames);
    for jj=1:valFrames
        temp = valMov(:,:,jj);
        if DIM1*DIM1 ~= DIM*DIM
            temp = imresize(temp,DIM/DIM1);
        end
        newValMov(:,:,jj) = temp - meanIm;
    end
    valMov = newValMov;clear newValMov;
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
    cellid = celldata(ii).cellid;
    [prediction] = DCTModel(mov,inds,DIM,numBack,numFrames,newResp,histDesign,train,valMov);
    save(sprintf('Cell%d_DCT.mat',ii),'prediction','cellid');
    
    [prediction] = WVLTModel(mov,inds,DIM,numBack,numFrames,newResp,histDesign,train,valMov);
    save(sprintf('Cell%d_WVLT.mat',ii),'prediction','cellid');
    
    [prediction] = DCTOASModel(mov,inds,DIM,numBack,numFrames,newResp,histDesign,train,valMov);
    save(sprintf('Cell%d_DCTOAS.mat',ii),'prediction','cellid');
    
    [prediction] = WVLTOASModel(mov,inds,DIM,numBack,numFrames,newResp,histDesign,train,valMov);
    save(sprintf('Cell%d_WVLTOAS.mat',ii),'prediction','cellid');
end
end

function [prediction] = DCTModel(mov,inds,DIM,numBack,numFrames,newResp,histDesign,train,valMov)
% DCT-PCA model
% dimRun = [90,80,70,60,50,40,30,20,10.*ones(1,numBack-8)];
dimRun = 30.*ones(1,numBack);
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

% [b] = SGD(histDesign(1:trainN,:),reduceDctData(1:trainN,1:Q),newResp(1:trainN));

tempDesign = reduceDctData(1:trainN,:);tempData = newResp(1:trainN);
% tempHist = histDesign(1:trainN,:);numHist = size(tempHist,2);
indsToKeep = round((0.1:0.1:1).*Q);
numInds = length(indsToKeep);

numIter = 1e3;

logMaxResp = max(log(newResp));
dataLen = length(newResp(trainN+1:end));
designLen = length(tempData);
heldOutDev = zeros(numInds,1);
keepN = round(0.1*designLen);
for ii=1:numInds
   guess = zeros(dataLen,numIter);
   for jj=1:numIter
       inds = random('Discrete Uniform',designLen,[keepN,1]);
       inds2 = randperm(Q,indsToKeep(ii));
       [b,~,~] = glmfit(tempDesign(inds,inds2),tempData(inds),'poisson');
       
%        startHist = tempHist(randperm(designLen,1),:);
%        temp = zeros(dataLen,1);
%        for kk=1:dataLen
%            temp(kk) = exp(b(1)+startHist*b(2:numHist+1)+reduceDctData(trainN+kk,inds2)*b(numHist+2:end));
%            startHist = [temp(kk),startHist(1:end-1)];
%        end
       guess(:,jj) = exp(b(1)+reduceDctData(trainN+1:end,inds2)*b(2:end));
   end
   temp = mean(guess,1);inds = find(temp>50);
   guess(:,inds) = [];
   if size(guess,1) ~= dataLen
       guess = mean(newResp(1:trainN)).*ones(dataLen,1);
   end
   predictTrain = min(mean(guess,2),exp(logMaxResp)+0.5);
   trueTrain = newResp(trainN+1:end);
   
   modelDev = GetDeviance(trueTrain,predictTrain);
   
   nullEst = mean(trueTrain).*ones(length(trueTrain),1);
   nullDev = GetDeviance(trueTrain,nullEst);
   
   heldOutDev(ii) = 1-modelDev/nullDev;
   corrcoef(predictTrain,trueTrain)
end

[maxDev,ind] = max(heldOutDev);
indsToKeep = indsToKeep(ind);
disp(ind);
disp(maxDev);

[~,~,numFrames] = size(valMov);
dctMov = zeros(numFrames,fullDctDim);

count = 1;
for jj=1:numFrames
    miniMov = zeros(DIM,DIM,numBack);
    for kk=1:numBack
        if jj-kk-2 < 1
            temp = zeros(DIM,DIM);
        else
            temp = valMov(:,:,max(jj-kk-2,1));
        end
        miniMov(:,:,kk) = temp;
    end
    R = mirt_dctn(miniMov);
    R = R(dctDims);
    dctMov(count,:) = R(:);
    count = count+1;
end

dctMov = dctMov';
dctMov = dctMov-repmat(mu,[1,numFrames]);

x = Winv*dctMov; % number of dimensions kept by N
reduceDctData2 = x';


numIter = 1e5;
designLen = length(reduceDctData);
keepN = round(0.1*designLen);

guess = zeros(numFrames,numIter);
for jj=1:numIter
    inds = random('Discrete Uniform',designLen,[keepN,1]);
    inds2 = randperm(Q,indsToKeep);
    [b,~,~] = glmfit(reduceDctData(inds,inds2),newResp(inds),'poisson');
    guess(:,jj) = exp(b(1)+reduceDctData2(:,inds2)*b(2:end));
end
temp = mean(guess,1);inds = find(temp>50);
guess(:,inds) = [];
prediction = min(mean(guess,2),exp(logMaxResp)+0.5);

end

function [dev] = GetDeviance(trueTrain,predictTrain)
initialDev = trueTrain.*log(trueTrain./predictTrain)-(trueTrain-predictTrain);
initialDev(isnan(initialDev) | isinf(initialDev)) = predictTrain(isnan(initialDev) | isinf(initialDev));
dev = 2*sum(initialDev);
end

% function [prediction] = GetModelFull(histDesign,reduceDesign,b,histParams)
% 
% prediction = exp(b(1)+histDesign*b(2:histParams+1)+reduceDesign*b(histParams+2:end));
% 
% end

function [prediction] = GetModelSmall(reduceDesign,b,histParams,maxLogResp)

prediction = exp(min(b(1)+reduceDesign*b(histParams+2:end),maxLogResp));

end

% function [b] = SGD(histDesign,reduceDesign,newResp)
% trainN = round(0.8*length(newResp));
% 
% [bInit,devInit,~] = glmfit([reduceDesign(1:trainN,:)],...
%     newResp(1:trainN),'poisson');
% 
% maxIter = 1e5;
% histParams = 0;%size(histDesign,2);
% lassoInds = histParams+2:length(bInit);
% 
% numParams = length(lassoInds);
% 
% onenorm = norm(bInit(lassoInds),1);
% penalty = ([0,1/100,1/10,0.5,1,2,3,4,5,7.5,10,15,25,50,100].*devInit)./onenorm;
% 
% numRuns = length(penalty);
% heldOutDev = zeros(numRuns,1);
% allB = cell(numRuns,1);
% 
% getObjective = @(b,dev,penalty) (dev+norm(b,1)*penalty);
% 
% maxLogResp = max(log(newResp));
% allB{1} = bInit;
% prediction = GetModelSmall(reduceDesign(trainN+1:end,:),bInit,histParams,maxLogResp);
% % figure;plot(newResp(trainN+1:end));hold on;plot(prediction);pause(0.1);
% heldOutDev(1) = GetDeviance(newResp(trainN+1:end),prediction);
% for ii=2:numRuns
%    lambda = penalty(ii);
%    
%    b = bInit;
%    prediction = GetModelSmall(reduceDesign(1:trainN,:),b,histParams,maxLogResp);
%    objective = getObjective(b(lassoInds),GetDeviance(newResp(1:trainN),prediction),lambda);
%    
%    iterB = zeros(maxIter,length(b));
%    iterObj = zeros(maxIter,1);
%    
%    iterObj(1) = objective;
%    iterB(1,:) = b;
%    for jj=2:maxIter
%        tempB = iterB(jj-1,:)';tempB2 = tempB(lassoInds);
%        inds = random('Discrete Uniform',numParams,[1,1]);
%        
%        tempB2(inds) = tempB2(inds) + normrnd(0,0.1);
%        tempB(lassoInds) = tempB2;
%        
%        prediction = GetModelSmall(reduceDesign(1:trainN,:),tempB,histParams,maxLogResp);
%        tempobjective = getObjective(tempB(lassoInds),GetDeviance(newResp(1:trainN),prediction),lambda);
%        
%        logA = iterObj(jj-1)-tempobjective;
%        if log(rand)<logA
%            iterObj(jj) = tempobjective;
%            iterB(jj,:) = tempB;
%        else
%            iterObj(jj) = iterObj(jj-1);
%            iterB(jj,:) = iterB(jj-1,:);
%        end
% %        plot(jj,iterObj(jj),'.');hold on;pause(1/1000);
%    end
%    [~,ind] = min(iterObj);
%    
%    b = iterB(ind,:)';allB{ii} = b;
%    prediction = GetModelSmall(reduceDesign(trainN+1:end,:),b,histParams,maxLogResp);
%    figure;plot(newResp(trainN+1:end));hold on;plot(prediction);pause(0.1);
%    heldOutDev(ii) = GetDeviance(newResp(trainN+1:end),prediction);
% end
% 
% [~,ind] = min(heldOutDev);
% b = allB{ind};
% 
% fprintf('Best Penalty: %3.1f\n',penalty(ind));
% end


function [prediction] = WVLTModel(mov,inds,DIM,numBack,numFrames,newResp,histDesign,train,valMov)
% wavelet-PCA model
miniMov = zeros(DIM,DIM,numBack);
R = wavedec3(miniMov,1,'db4');
R = R.dec;R = R{1};

fullWVLTdim = numel(R)*4;
wvltMov = zeros(numFrames,fullWVLTdim);

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

maxToKeep = 3.5e4;p = maxToKeep/fullWVLTdim;
temp = var(wvltMov,[],1);
Q = quantile(temp,1-p);
tempInds = temp>=Q;
wvltMov = wvltMov(:,tempInds);
fullWVLTdim2 = size(wvltMov,2);
S = cov(wvltMov); % or try shrinkage_cov
% S = shrinkage_cov(dctMov);
[V,D] = eig(S);clear S;

wvltMov = wvltMov';
mu = mean(wvltMov,2);
wvltMov = wvltMov-repmat(mu,[1,numFrames]);

allEigs = diag(D);
fullVariance = sum(allEigs);
for jj=50:1:fullWVLTdim2-10
    start = fullWVLTdim2-jj+1;
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

trainN = round(length(newResp).*train);

tempDesign = reduceWvltData(1:trainN,:);tempData = newResp(1:trainN);
% tempHist = histDesign(1:trainN,:);
indsToKeep = round((0.1:0.1:1).*Q);
numInds = length(indsToKeep);

numIter = 1e4;

logMaxResp = max(log(newResp));
dataLen = length(newResp(trainN+1:end));
designLen = length(tempData);
heldOutDev = zeros(numInds,1);
keepN = round(0.1*designLen);
for ii=1:numInds
   guess = zeros(dataLen,numIter);
   for jj=1:numIter
       inds = random('Discrete Uniform',designLen,[keepN,1]);
       inds2 = random('Discrete Uniform',Q,[indsToKeep(ii),1]);
       [b,~,~] = glmfit(tempDesign(inds,inds2),tempData(inds),'poisson');
       guess(:,jj) = exp(b(1)+reduceWvltData(trainN+1:end,inds2)*b(2:end));
   end
   predictTrain = min(mean(guess,2),exp(logMaxResp)+0.5);
   trueTrain = newResp(trainN+1:end);
   
   modelDev = GetDeviance(trueTrain,predictTrain);
   
   nullEst = mean(trueTrain).*ones(length(trueTrain),1);
   nullDev = GetDeviance(trueTrain,nullEst);
   
   heldOutDev(ii) = 1-modelDev/nullDev;
end

[maxDev,ind] = max(heldOutDev);
indsToKeep = indsToKeep(ind);
disp(ind);
disp(maxDev);

[~,~,numFrames] = size(valMov);

wvltMov = zeros(numFrames,fullWVLTdim);

count = 1;
for jj=1:numFrames
    miniMov = zeros(DIM,DIM,numBack);
    for kk=1:numBack
        if jj-kk-2 < 1
            temp = zeros(DIM,DIM);
        else
            temp = valMov(:,:,max(jj-kk-2,1));
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

wvltMov = wvltMov(:,tempInds);

wvltMov = wvltMov';
wvltMov = wvltMov-repmat(mu,[1,numFrames]);

x = Winv*wvltMov; % number of dimensions kept by N
reduceWvltData2 = x';


numIter = 1e5;
designLen = length(reduceWvltData);
keepN = round(0.1*designLen);

guess = zeros(numFrames,numIter);
for jj=1:numIter
    inds = random('Discrete Uniform',designLen,[keepN,1]);
    inds2 = random('Discrete Uniform',Q,[indsToKeep,1]);
    [b,~,~] = glmfit(reduceWvltData(inds,inds2),newResp(inds),'poisson');
    guess(:,jj) = exp(b(1)+reduceWvltData2(:,inds2)*b(2:end));
end
prediction = min(mean(guess,2),exp(logMaxResp)+0.5);

end

function [prediction] = DCTOASModel(mov,inds,DIM,numBack,numFrames,newResp,histDesign,train,valMov)
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

tempDesign = reduceDctData(1:trainN,:);tempData = newResp(1:trainN);
% tempHist = histDesign(1:trainN,:);
indsToKeep = round((0.1:0.1:1).*Q);
numInds = length(indsToKeep);

numIter = 1e4;

logMaxResp = max(log(newResp));
dataLen = length(newResp(trainN+1:end));
designLen = length(tempData);
heldOutDev = zeros(numInds,1);
keepN = round(0.1*designLen);
for ii=1:numInds
   guess = zeros(dataLen,numIter);
   for jj=1:numIter
       inds = random('Discrete Uniform',designLen,[keepN,1]);
       inds2 = random('Discrete Uniform',Q,[indsToKeep(ii),1]);
       [b,~,~] = glmfit(tempDesign(inds,inds2),tempData(inds),'poisson');
       guess(:,jj) = exp(b(1)+reduceDctData(trainN+1:end,inds2)*b(2:end));
   end
   predictTrain = min(mean(guess,2),exp(logMaxResp)+0.5);
   trueTrain = newResp(trainN+1:end);
   
   modelDev = GetDeviance(trueTrain,predictTrain);
   
   nullEst = mean(trueTrain).*ones(length(trueTrain),1);
   nullDev = GetDeviance(trueTrain,nullEst);
   
   heldOutDev(ii) = 1-modelDev/nullDev;
end

[maxDev,ind] = max(heldOutDev);
indsToKeep = indsToKeep(ind);
disp(ind);
disp(maxDev);

[~,~,numFrames] = size(valMov);
dctMov = zeros(numFrames,fullDctDim);

count = 1;
for jj=1:numFrames
    miniMov = zeros(DIM,DIM,numBack);
    for kk=1:numBack
        if jj-kk-2 < 1
            temp = zeros(DIM,DIM);
        else
            temp = valMov(:,:,max(jj-kk-2,1));
        end
        miniMov(:,:,kk) = temp;
    end
    R = mirt_dctn(miniMov);
    R = R(dctDims);
    dctMov(count,:) = R(:);
    count = count+1;
end

dctMov = dctMov';
dctMov = dctMov-repmat(mu,[1,numFrames]);

x = Winv*dctMov; % number of dimensions kept by N
reduceDctData2 = x';


numIter = 1e5;
designLen = length(reduceDctData);
keepN = round(0.1*designLen);

guess = zeros(numFrames,numIter);
for jj=1:numIter
    inds = random('Discrete Uniform',designLen,[keepN,1]);
    inds2 = random('Discrete Uniform',Q,[indsToKeep,1]);
    [b,~,~] = glmfit(reduceDctData(inds,inds2),newResp(inds),'poisson');
    guess(:,jj) = exp(b(1)+reduceDctData2(:,inds2)*b(2:end));
end
prediction = min(mean(guess,2),exp(logMaxResp)+0.5);

end

function [prediction] = WVLTOASModel(mov,inds,DIM,numBack,numFrames,newResp,histDesign,train,valMov)
% wavelet-PCA model
miniMov = zeros(DIM,DIM,numBack);
R = wavedec3(miniMov,1,'db4');
R = R.dec;R = R{1};

fullWVLTdim = numel(R)*4;
wvltMov = zeros(numFrames,fullWVLTdim);

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

maxToKeep = 3.5e4;p = maxToKeep/fullWVLTdim;
temp = var(wvltMov,[],1);
Q = quantile(temp,1-p);
tempInds = temp>=Q;
wvltMov = wvltMov(:,tempInds);
fullWVLTdim2 = size(wvltMov,2);
% S = cov(wvltMov); % or try shrinkage_cov
S = shrinkage_cov(wvltMov);
[V,D] = eig(S);clear S;

wvltMov = wvltMov';
mu = mean(wvltMov,2);
wvltMov = wvltMov-repmat(mu,[1,numFrames]);

allEigs = diag(D);
fullVariance = sum(allEigs);
for jj=50:1:fullWVLTdim2-10
    start = fullWVLTdim2-jj+1;
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

trainN = round(length(newResp).*train);

tempDesign = reduceWvltData(1:trainN,:);tempData = newResp(1:trainN);
% tempHist = histDesign(1:trainN,:);
indsToKeep = round((0.1:0.1:1).*Q);
numInds = length(indsToKeep);

numIter = 1e4;

logMaxResp = max(log(newResp));
dataLen = length(newResp(trainN+1:end));
designLen = length(tempData);
heldOutDev = zeros(numInds,1);
keepN = round(0.1*designLen);
for ii=1:numInds
   guess = zeros(dataLen,numIter);
   for jj=1:numIter
       inds = random('Discrete Uniform',designLen,[keepN,1]);
       inds2 = random('Discrete Uniform',Q,[indsToKeep(ii),1]);
       [b,~,~] = glmfit(tempDesign(inds,inds2),tempData(inds),'poisson');
       guess(:,jj) = exp(b(1)+reduceWvltData(trainN+1:end,inds2)*b(2:end));
   end
   predictTrain = min(mean(guess,2),exp(logMaxResp)+0.5);
   trueTrain = newResp(trainN+1:end);
   
   modelDev = GetDeviance(trueTrain,predictTrain);
   
   nullEst = mean(trueTrain).*ones(length(trueTrain),1);
   nullDev = GetDeviance(trueTrain,nullEst);
   
   heldOutDev(ii) = 1-modelDev/nullDev;
end

[maxDev,ind] = max(heldOutDev);
indsToKeep = indsToKeep(ind);
disp(ind);
disp(maxDev);

[~,~,numFrames] = size(valMov);

wvltMov = zeros(numFrames,fullWVLTdim);

count = 1;
for jj=1:numFrames
    miniMov = zeros(DIM,DIM,numBack);
    for kk=1:numBack
        if jj-kk-2 < 1
            temp = zeros(DIM,DIM);
        else
            temp = valMov(:,:,max(jj-kk-2,1));
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

wvltMov = wvltMov(:,tempInds);

wvltMov = wvltMov';
wvltMov = wvltMov-repmat(mu,[1,numFrames]);

x = Winv*wvltMov; % number of dimensions kept by N
reduceWvltData2 = x';


numIter = 1e5;
designLen = length(reduceWvltData);
keepN = round(0.1*designLen);

guess = zeros(numFrames,numIter);
for jj=1:numIter
    inds = random('Discrete Uniform',designLen,[keepN,1]);
    inds2 = random('Discrete Uniform',Q,[indsToKeep,1]);
    [b,~,~] = glmfit(reduceWvltData(inds,inds2),newResp(inds),'poisson');
    guess(:,jj) = exp(b(1)+reduceWvltData2(:,inds2)*b(2:end));
end
prediction = min(mean(guess,2),exp(logMaxResp)+0.5);

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