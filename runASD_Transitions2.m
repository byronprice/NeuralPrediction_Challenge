cellinfo

numCells = length(celldata);

prediction = struct('cellid',cell(numCells,1),'response',zeros(713,1));
%rfs = cell(numCells,3);

neuralResponse = cell(numCells,1);
for ii=1:numCells
    load(celldata(ii).datafile,'resp');
    neuralResponse{ii} = resp;
end
    
for ii=1:numCells
    resp = neuralResponse{ii};
    inds = find(~isnan(resp));
    newResp = resp(~isnan(resp));
    numFrames = length(newResp);
    mov = loadimfile(celldata(ii).fullstimfile);
    
    meanResponse = mean(newResp);
    newResp = newResp-meanResponse;
    
    [~,~,trueFrames] = size(mov);
    corrs = zeros(trueFrames,1);
    for jj=2:trueFrames
        prevFrame = mov(:,:,jj-1);
        currentFrame = mov(:,:,jj);
        r = corrcoef(prevFrame(:),currentFrame(:));
        corrs(jj) = r(1,2);
    end
    transitionInds = corrs<0.95;
    
    numBack = 30;Design = zeros(trueFrames,numBack);
    temp = double(transitionInds);
    for jj=0:numBack-1
        forcorr = temp(1:end-jj);
        Design(:,jj+1) = forcorr;
        temp = [0;temp];
    end
    Design = Design(inds,2:end);
    [~,timePoints] = size(Design);
    
    numBases = 10;
    stdevs = 0.5:0.1:8;
    numIter = 250;
    holdOutDev = zeros(length(stdevs),1);
    allInds = 1:numFrames;cvLen = round(numFrames*0.8);
    for jj=1:length(stdevs)
        basisFuns = zeros(timePoints,numBases);
        time = linspace(0,timePoints-1,timePoints);
        centerPoints = linspace(0,timePoints-1,numBases);
        for kk=1:numBases
            basisFuns(:,kk) = exp(-(time-centerPoints(kk)).^2./(2*stdevs(jj)*stdevs(jj)));
        end
        deviance = zeros(numIter,1);
        for kk=1:numIter
            tempInds = randperm(numFrames,cvLen);
            tempInds = ismember(allInds,tempInds);
            holdOutInds = ~tempInds;
            [b,~,~] = glmfit(Design(tempInds,:)*basisFuns,newResp(tempInds)+meanResponse,'poisson');
            estimate = exp(b(1)+Design(holdOutInds,:)*basisFuns*b(2:end));
            y = newResp(holdOutInds)+meanResponse;
            initialDev = y.*log(y./estimate)-(y-estimate);
            initialDev(isnan(initialDev) | isinf(initialDev)) = estimate(isnan(initialDev) | isinf(initialDev));
            deviance(kk) = 2*sum(initialDev);
        end
        holdOutDev(jj) = median(deviance);
    end
    [~,ind] = min(holdOutDev);
    stdev = stdevs(ind);
    basisFuns = zeros(timePoints,numBases);
    time = linspace(0,timePoints-1,timePoints);
    centerPoints = linspace(0,timePoints-1,numBases);
    for kk=1:numBases
        basisFuns(:,kk) = exp(-(time-centerPoints(kk)).^2./(2*stdev*stdev));
    end

    [b,minDev,~] = glmfit(Design*basisFuns,newResp+meanResponse,'poisson');
    [~,dev,~] = glmfit(ones(length(newResp),1),newResp+meanResponse,'poisson');
    
    fprintf('\nOptimal Standard Deviation: %3.3f\n',stdev);
    fprintf('\nExplained Deviance: %3.3f\n',1-minDev/dev);
    
    fprintf('\nAIC_Null: %3.3f  AIC_Model: %3.3f\n',2*1-dev,2*(1+numBases)-minDev);
    
    mov = loadimfile(celldata(ii).fullvalstimfile);
    
    prediction(ii).cellid = celldata(ii).cellid;
    
    [~,~,trueFrames] = size(mov);
    corrs = zeros(trueFrames,1);
    for jj=2:trueFrames
        prevFrame = mov(:,:,jj-1);
        currentFrame = mov(:,:,jj);
        r = corrcoef(prevFrame(:),currentFrame(:));
        corrs(jj) = r(1,2);
    end
    transitionInds = corrs<0.95;
    
    Design = zeros(trueFrames,numBack);
    temp = double(transitionInds);
    for jj=0:numBack-1
        forcorr = temp(1:end-jj);
        Design(:,jj+1) = forcorr;
        temp = [0;temp];
    end
    Design = Design(:,2:end);
    
    prediction(ii).response = exp(b(1)+Design*basisFuns*b(2:end));
end

save('ASD_Predictions_Transitions.mat','prediction');
% save('ASD_Transitions.mat','rfs');