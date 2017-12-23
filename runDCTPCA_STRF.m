% runDCTPCA_STRF.m
warning('off','all');

cellinfo

numCells = length(celldata);

prediction1 = struct('cellid',cell(numCells,1),'response',zeros(713,1));
prediction2 = struct('cellid',cell(numCells,1),'response',zeros(713,1));
prediction3 = struct('cellid',cell(numCells,1),'response',zeros(713,1));
prediction4 = struct('cellid',cell(numCells,1),'response',zeros(713,1));
rfs = struct('RFdev',cell(numCells,1),'RFaic',cell(numCells,1),'ToSubtract',cell(numCells,1),...
    'CorrDev',cell(numCells,1),'MeanFiring',cell(numCells,1),'devInds',cell(numCells,1),...
    'CorrAIC',cell(numCells,1),'aicInds',cell(numCells,1));

neuralResponse = cell(numCells,1);
for ii=1:numCells
    load(celldata(ii).datafile,'resp');
    neuralResponse{ii} = resp;
end

% myCluster = parcluster('local');
% 
% if getenv('ENVIRONMENT')
%    myCluster.JobStorageLocation = getenv('TMPDIR');
% end
% 
% parpool(myCluster,12);

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
    
    responseMean = mean(newResp);
    newResp = newResp-mean(newResp);
   
    numBack = 30;
    
    % get transition and history model / basis functions
    corrs = zeros(fullFrames,1);
    for jj=2:fullFrames
        prevFrame = mov(:,:,jj-1);
        currentFrame = mov(:,:,jj);
        r = corrcoef(prevFrame(:),currentFrame(:));
        corrs(jj) = r(1,2);
    end
    transitionInds = corrs<0.95;
    transDesign = zeros(fullFrames,numBack);
    temp = double(transitionInds);
    for jj=0:numBack-1
        forcorr = temp(1:end-jj);
        transDesign(:,jj+1) = forcorr;
        temp = [0;temp];
    end
    
    transDesign = transDesign(inds,2:end);
    [~,timePoints] = size(transDesign);
    
    transBases = 10;histBases = 10;
    transdev = 2;histdev = 1.75;
    
    histBasisFuns = zeros(histLags,histBases);
    time = linspace(0,histLags-1,histLags);
    centerPoints = linspace(0,histLags-1,histBases);
    for kk=1:histBases
        histBasisFuns(:,kk) = exp(-(time-centerPoints(kk)).^2./(2*histdev*histdev));
    end
    
    transBasisFuns = zeros(timePoints,transBases);
    time = linspace(0,timePoints-1,timePoints);
    centerPoints = linspace(0,timePoints-1,transBases);
    for kk=1:transBases
        transBasisFuns(:,kk) = exp(-(time-centerPoints(kk)).^2./(2*transdev*transdev));
    end
    
    dimRun = [90,80,70,60,50,40,30,20,10,ones(1,numBack-8).*5];
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
    for jj=100:1:fullDctDim-10
        start = fullDctDim-jj+1;
        eigenvals = allEigs(start:end);
        varianceProp = sum(eigenvals)/fullVariance;
        if varianceProp >= 0.995
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
    
    [fullB,~,~] = glmfit([transDesign*transBasisFuns,histDesign*histBasisFuns,...
        reduceDctData],newResp+responseMean,'poisson');
    
    transInds = 2:transBases+1;histInds = transBases+2:length(fullB)-Q;
    pcaInds = (length(fullB)-Q+1):1:length(fullB);
    pcaB = fullB(pcaInds);
   
    indsToKeep = (1:5:Q)./Q;indLen = length(indsToKeep);
    holdOutDev = zeros(indLen,2);
    for yy=1:indLen
        myquant = quantile(abs(pcaB),1-indsToKeep(yy));
        newBinds = abs(pcaB)>=myquant;
        
        numIter = 250;
        train = round(numFrames.*0.7);test = numFrames-train;
        tempHoldOut = zeros(numIter,2);
        allInds = 1:numFrames;
        for zz=1:numIter
            inds = randperm(numFrames,train);
            inds = ismember(allInds,inds);
            holdOutInds = ~inds;
            
           % estFilt = reduceDctData(inds,:)\newResp(inds);
            [b,~,~] = glmfit([transDesign(inds,:)*transBasisFuns,histDesign(inds,:)*histBasisFuns,...
                reduceDctData(inds,newBinds)],newResp(inds)+responseMean,'poisson');
            
            y = newResp(holdOutInds)+responseMean;
            
            estimate = exp(b(1)+transDesign(holdOutInds,:)*transBasisFuns*b(transInds)+...
                histDesign(holdOutInds,:)*histBasisFuns*b(histInds)+...
                reduceDctData(holdOutInds,newBinds)*b(histInds(end)+1:end));
            initialDev = y.*log(y./estimate)-(y-estimate);
            initialDev(isnan(initialDev) | isinf(initialDev)) = estimate(isnan(initialDev) | isinf(initialDev));
            modelDev = 2*sum(initialDev);
            tempHoldOut(zz,1) = modelDev;
            tempHoldOut(zz,2) = 2*length(b)+modelDev;
        end
        holdOutDev(yy,1) = median(tempHoldOut(:,1));
        holdOutDev(yy,2) = median(tempHoldOut(:,2));
    end
    [~,ind1] = min(holdOutDev(:,1));
    [~,ind2] = min(holdOutDev(:,2));
    myquant = quantile(abs(pcaB),1-indsToKeep(ind1));
    newBinds1 = abs(pcaB)>myquant;
    
    myquant = quantile(abs(pcaB),1-indsToKeep(ind2));
    newBinds2 = abs(pcaB)>myquant;
    
    [b1,~,~] = glmfit([transDesign*transBasisFuns,histDesign*histBasisFuns,...
                reduceDctData(:,newBinds1)],newResp+responseMean,'poisson');
            
    [b2,~,~] = glmfit([transDesign*transBasisFuns,histDesign*histBasisFuns,...
                reduceDctData(:,newBinds2)],newResp+responseMean,'poisson');
    
    estimate = exp(b1(1)+transDesign*transBasisFuns*b1(transInds)+...
                histDesign*histBasisFuns*b1(histInds)+reduceDctData(:,newBinds1)*b1(histInds(end)+1:end));
    r1 = corrcoef(estimate,newResp+responseMean);
    
    estimate = exp(b2(1)+transDesign*transBasisFuns*b2(transInds)+...
                histDesign*histBasisFuns*b2(histInds)+reduceDctData(:,newBinds2)*b2(histInds(end)+1:end));
    r2 = corrcoef(estimate,newResp+responseMean);
    
    fprintf('\nCorrelation Dev: %3.3f\n\n\n',r1(1,2));
    fprintf('\nCorrelation AIC: %3.3f\n\n\n',r2(1,2));
    pause(1);
    
    rfs(ii).RFdev = b1;
    rfs(ii).RFaic = b2;
    rfs(ii).MeanFiring = responseMean;
    rfs(ii).ToSubtract = mu;
    rfs(ii).CorrDev = r1(1,2);
    rfs(ii).CorrAIC = r2(1,2);
    rfs(ii).devInds = newBinds1;
    rfs(ii).aicInds = newBinds2;
    
    % validation set
    mov = loadimfile(celldata(ii).fullvalstimfile);
    [DIM1,DIM2,numFrames] = size(mov);
    
    meanIm = imresize(meanIm,DIM1/DIM);
    for jj=1:numFrames
        mov(:,:,jj) = mov(:,:,jj) - meanIm;
    end
    prediction1(ii).cellid = celldata(ii).cellid;
    prediction2(ii).cellid = celldata(ii).cellid;
    prediction3(ii).cellid = celldata(ii).cellid;
    prediction4(ii).cellid = celldata(ii).cellid;
    
    dctMov = zeros(numFrames,fullDctDim);
    
    miniMov = zeros(DIM,DIM,numBack);
    for jj=1:numFrames
        for kk=1:numBack
            if jj-kk-2 < 1
                temp = zeros(DIM1,DIM2);
            else
                temp = mov(:,:,max(jj-kk-2,1));
            end
            if DIM1*DIM2 ~= DIM*DIM
                temp = imresize(temp,DIM/DIM1);
            end
            miniMov(:,:,kk) = temp;
        end
        R = mirt_dctn(miniMov);
        R = R(dctDims);
        dctMov(jj,:) = R(:);
    end
    dctMov = dctMov';
  %  mu2 = mean(dctMov,2);
    dctMov = dctMov-repmat(mu,[1,numFrames]);
    x = Winv*dctMov;
    reduceDctData = x'; % N-by
    
    [~,~,trueFrames] = size(mov);
    corrs = zeros(trueFrames,1);
    for jj=2:trueFrames
        prevFrame = mov(:,:,jj-1);
        currentFrame = mov(:,:,jj);
        r = corrcoef(prevFrame(:),currentFrame(:));
        corrs(jj) = r(1,2);
    end
    transitionInds = corrs<0.95;
    
    transDesign = zeros(trueFrames,numBack);
    temp = double(transitionInds);
    for jj=0:numBack-1
        forcorr = temp(1:end-jj);
        transDesign(:,jj+1) = forcorr;
        temp = [0;temp];
    end
    transDesign = transDesign(:,2:end);
            
    numIter = 5000;
    estResponse1 = zeros(trueFrames,numIter);
    estResponse2 = zeros(trueFrames,numIter);
    for jj=1:numIter
        currentHist1 = poissrnd(exp(b1(1)),[1,histLags]);
        currentHist2 = poissrnd(exp(b2(1)),[1,histLags]);
        for kk=1:trueFrames
            estResponse1(kk,jj) = poissrnd(exp(b1(1)+transDesign(kk,:)*transBasisFuns*b1(transInds)+...
                currentHist1*histBasisFuns*b1(histInds)+reduceDctData(kk,newBinds1)*b1(histInds(end)+1:end)));
            currentHist1 = [estResponse1(kk,jj),currentHist1(1:end-1)];
            
            estResponse2(kk,jj) = poissrnd(exp(b2(1)+transDesign(kk,:)*transBasisFuns*b2(transInds)+...
                currentHist2*histBasisFuns*b2(histInds)+reduceDctData(kk,newBinds2)*b2(histInds(end)+1:end)));
            currentHist2 = [estResponse2(kk,jj),currentHist2(1:end-1)];
        end
    end
    
    prediction1(ii).response = median(estResponse1,2);
    prediction2(ii).response = median(estResponse2,2);
    prediction3(ii).response = exp(b1(1)+transDesign*transBasisFuns*b1(transInds)+...
                reduceDctData(:,newBinds1)*b1(histInds(end)+1:end));
    prediction4(ii).response = exp(b2(1)+transDesign*transBasisFuns*b2(transInds)+...
                reduceDctData(:,newBinds2)*b2(histInds(end)+1:end));
end

save('ASD_PredictionsSTRF_DCTPCA.mat','prediction1','prediction2','prediction3','prediction4');
save('ASD_STRFs_DCTPCA.mat','rfs');

% delete(gcp);
