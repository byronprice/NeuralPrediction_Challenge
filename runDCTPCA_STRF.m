% runDCTPCA_STRF.m

cellinfo

numCells = length(celldata);

prediction = struct('cellid',cell(numCells,1),'response',zeros(713,1));
rfs = struct('RF',cell(numCells,1),'ToSubtract',cell(numCells,1),...
    'Corr',cell(numCells,1),'MeanFiring',cell(numCells,1),'prefDIM',cell(numCells,1));

neuralResponse = cell(numCells,1);
for ii=1:numCells
    load(celldata(ii).datafile,'resp');
    neuralResponse{ii} = resp;
end

myCluster = parcluster('local');

if getenv('ENVIRONMENT')
   myCluster.JobStorageLocation = getenv('TMPDIR');
end

parpool(myCluster,12);

parfor ii=1:numCells
    resp = neuralResponse{ii};
    inds = find(~isnan(resp));
    newResp = resp(~isnan(resp));
    numFrames = length(newResp);
    mov = loadimfile(celldata(ii).fullstimfile);
    
    [DIM,~,fullFrames] = size(mov);
    meanIm = mean(mov,3);
    for jj=1:fullFrames
        mov(:,:,jj) = mov(:,:,jj) - meanIm;
    end
    
    responseMean = mean(newResp);
    newResp = newResp-mean(newResp);
   
    numBack = 25;
    
    dctDim = [50,50,6];
    fullDctDim = prod(dctDim);
    dctMov = zeros(numFrames,fullDctDim);
    
    centerMean = mean(mean(mean(mov(20:50,20:50,:))));
    count = 1;
    for jj=inds'
        miniMov = zeros(DIM,DIM,numBack);
        for kk=1:numBack
            if jj-kk-2 < 1
                temp = centerMean.*ones(DIM,DIM);
            else
                temp = mov(:,:,max(jj-kk-2,1));
            end
            miniMov(:,:,kk) = temp;
        end
        R = mirt_dctn(miniMov);
        R = R(1:dctDim(1),1:dctDim(2),1:dctDim(3));
        dctMov(count,:) = R(:);
        count = count+1;
    end
    
    S = cov(dctMov); % or try shrinkage_cov
    % S = shrinkage_cov(dctMov);
    [V,D] = eig(S);
    
    dctMov = dctMov';
    mu = mean(dctMov,2);
    dctMov = dctMov-repmat(mu,[1,numFrames]);
    
    saveVar = 0.7:0.01:0.99;saveVar = [saveVar,0.991:0.001:0.995];
    varLen = length(saveVar);
    holdOutCorr = zeros(varLen,2);
    
    for yy=1:varLen
        % choose dimensionality for dct images
        allEigs = diag(D);
        fullVariance = sum(allEigs);
        for jj=5:1:fullDctDim-10
            start = fullDctDim-jj+1;
            eigenvals = allEigs(start:end);
            varianceProp = sum(eigenvals)/fullVariance;
            if varianceProp >= saveVar(yy)
                break;
            end
        end
        
        q = jj;
        holdOutCorr(yy,2) = q;
        meanEig = mean(allEigs(1:start-1));
        W = V(:,start:end)*sqrtm(D(start:end,start:end)-meanEig.*eye(q));
        W = fliplr(W);
       
        x = pinv(W)*dctMov; % number of dimensions kept by N
        
        reduceDctData = x'; % N-by-however-many-dimensions are kept
        
        numIter = 5;
        train = round(numFrames.*0.8);test = numFrames-train;
        tempHoldOut = zeros(numIter,1);
        allInds = 1:numFrames;
        for zz=1:numIter
            inds = randperm(numFrames,train);
            inds = ismember(allInds,inds);
            holdOutInds = ~inds;
            
           % estFilt = reduceDctData(inds,:)\newResp(inds);
            [b,~,~] = glmfit(reduceDctData(inds,:),newResp(inds)+responseMean,'poisson');
            
            y = newResp(holdOutInds)+responseMean;
            nullEst = ones(test,1).*mean(y);
            initialDev = y.*log(y./nullEst)-(y-nullEst);
            initialDev(isnan(initialDev) | isinf(initialDev)) = nullEst(isnan(initialDev) | isinf(initialDev));
            nullDev = 2*sum(initialDev);
            
            estimate = exp(b(1)+reduceDctData(holdOutInds,:)*b(2:end));
            initialDev = y.*log(y./estimate)-(y-estimate);
            initialDev(isnan(initialDev) | isinf(initialDev)) = estimate(isnan(initialDev) | isinf(initialDev));
            modelDev = 2*sum(initialDev);
            tempHoldOut(zz) = 1-modelDev/nullDev;
        end
        holdOutCorr(yy,1) = median(tempHoldOut);
    end
    [~,ind] = max(holdOutCorr(:,1));
    prefDim = holdOutCorr(ind,2);
    
    start = fullDctDim-prefDim+1;
    meanEig = mean(allEigs(1:start-1));
    W = V(:,start:end)*sqrtm(D(start:end,start:end)-meanEig.*eye(prefDim));
    W = fliplr(W);
    Winv = pinv(W);
    x = Winv*dctMov;
    
    reduceDctData = x'; % N-by
    
    [b,~,~] = glmfit(reduceDctData,newResp+responseMean,'poisson');
    r = corrcoef(exp(b(1)+reduceDctData*b(2:end)),newResp);
    
    fprintf('\nCorrelation: %3.3f\n\n\n',r(1,2));
    pause(1);
    
    rfs(ii).RF = b;
    rfs(ii).MeanFiring = responseMean;
    rfs(ii).ToSubtract = mu;
    rfs(ii).Corr = r(1,2);
    rfs(ii).prefDIM = prefDim;
    
    % validation set
    mov = loadimfile(celldata(ii).fullvalstimfile);
    [DIM1,DIM2,numFrames] = size(mov);
    
    meanIm = imresize(meanIm,DIM1/DIM);
    for jj=1:numFrames
        mov(:,:,jj) = mov(:,:,jj) - meanIm;
    end
    prediction(ii).cellid = celldata(ii).cellid;
    
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
        R = R(1:dctDim(1),1:dctDim(2),1:dctDim(3));
        dctMov(jj,:) = R(:);
    end
    dctMov = dctMov';
  %  mu2 = mean(dctMov,2);
    dctMov = dctMov-repmat(mu,[1,numFrames]);
    x = Winv*dctMov;
    reduceDctData = x'; % N-by
    prediction(ii).response = exp(b(1)+reduceDctData*b(2:end));
end

save('ASD_PredictionsSTRF_DCTPCA.mat','prediction');
save('ASD_STRFs_DCTPCA.mat','rfs');

delete(gcp);
