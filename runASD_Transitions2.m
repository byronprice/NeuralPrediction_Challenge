cellinfo

numCells = length(celldata);
fprintf('Number of Cells: %d\n',numCells);
prediction = struct('cellid',cell(numCells,1),'response',zeros(713,1));
prediction2 = struct('cellid',cell(numCells,1),'response',zeros(713,1));
rfs = struct('History',cell(numCells,1),'Transition',cell(numCells,1));

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
    
    [DIM,~,trueFrames] = size(mov);
    convertMov = zeros(trueFrames,DIM*DIM);
    meanIm = mean(mov,3);
    for jj=1:trueFrames
        temp = mov(:,:,jj)-meanIm;
        convertMov(jj,:) = temp(:);
        mov(:,:,jj) = temp;
    end
    meanResponse = mean(newResp);
    newResp = newResp-meanResponse;
    
    corrs = zeros(trueFrames,1);
    for jj=2:trueFrames
        prevFrame = mov(:,:,jj-1);
        currentFrame = mov(:,:,jj);
        r = corrcoef(prevFrame(:),currentFrame(:));
        corrs(jj) = r(1,2);
    end
    transitionInds = corrs<0.95;
    
    numBack = 30;transDesign = zeros(trueFrames,numBack);
    temp = double(transitionInds);
    for jj=0:numBack-1
        forcorr = temp(1:end-jj);
        transDesign(:,jj+1) = forcorr;
        %r = corrcoef(newResp,forcorr(inds));
        temp = [0;temp];
    end
    
    % get spatial rf
%     [~,lagInd] = max(abs(corrPattern));
%     temp = double(transitionInds);
%     maxLagTransTimes = temp(1:end-lagInd+1);
%     maxLagTransTimes = maxLagTransTimes(inds);
%     convertMov = convertMov(inds,:);
%     [kest,~,~] = fastASD(convertMov(maxLagTransTimes,:),newResp(maxLagTransTimes),[DIM,DIM],0);
    
    transDesign = transDesign(inds,2:end);
    [~,timePoints] = size(transDesign);
    
    transBases = 10;histBases = 10;
    tranStdevs = 1:0.1:4;histStdevs = 1:0.1:4;
    numIter = 200;
    holdOutDev = zeros(length(tranStdevs),length(histStdevs));
    allInds = 1:numFrames;cvLen = round(numFrames*0.7);
    for jj=1:length(tranStdevs)
        transbasisFuns = zeros(timePoints,transBases);
        time = linspace(0,timePoints-1,timePoints);
        centerPoints = linspace(0,timePoints-1,transBases);
        for kk=1:transBases
            transbasisFuns(:,kk) = exp(-(time-centerPoints(kk)).^2./(2*tranStdevs(jj)*tranStdevs(jj)));
        end
        
        for ll=1:length(histStdevs)
            histbasisFuns = zeros(histLags,histBases);
            time = linspace(0,histLags-1,histLags);
            centerPoints = linspace(0,histLags-1,histBases);
            for kk=1:histBases
                histbasisFuns(:,kk) = exp(-(time-centerPoints(kk)).^2./(2*histStdevs(ll)*histStdevs(ll)));
            end
            
            deviance = zeros(numIter,1);
            for kk=1:numIter
                tempInds = randperm(numFrames,cvLen);
                tempInds = ismember(allInds,tempInds);
                holdOutInds = ~tempInds;
                [b,~,~] = glmfit([transDesign(tempInds,:)*transbasisFuns,...
                    histDesign(tempInds,:)*histbasisFuns],newResp(tempInds)+meanResponse,'poisson');
                
                estimate = exp(b(1)+transDesign(holdOutInds,:)*transbasisFuns*b(2:transBases+1)+...
                    histDesign(holdOutInds,:)*histbasisFuns*b(transBases+2:end));
                
                y = newResp(holdOutInds)+meanResponse;
                
                initialDev = y.*log(y./estimate)-(y-estimate);
                initialDev(isnan(initialDev) | isinf(initialDev)) = estimate(isnan(initialDev) | isinf(initialDev));
                deviance(kk) = 2*sum(initialDev);
            end
        holdOutDev(jj,ll) = median(deviance);
        end
    end
    [~,ind] = min(holdOutDev(:));
    [ind_row,ind_col] = ind2sub(size(holdOutDev),ind);
    stdev = tranStdevs(ind_row)+0.1;
    histstdev = histStdevs(ind_col)+0.1;
    rfs(ii).History = histstdev;
    rfs(ii).Transition = stdev;
    transbasisFuns = zeros(timePoints,transBases);
    time = linspace(0,timePoints-1,timePoints);
    centerPoints = linspace(0,timePoints-1,transBases);
    for kk=1:transBases
        transbasisFuns(:,kk) = exp(-(time-centerPoints(kk)).^2./(2*stdev*stdev));
    end
    
    histbasisFuns = zeros(histLags,histBases);
    time = linspace(0,histLags-1,histLags);
    centerPoints = linspace(0,histLags-1,histBases);
    for kk=1:histBases
        histbasisFuns(:,kk) = exp(-(time-centerPoints(kk)).^2./(2*histstdev*histstdev));
    end
    fprintf('\nDiscovered appropriate standard deviations\n');
    [b,minDev,~] = glmfit([transDesign*transbasisFuns,histDesign*histbasisFuns]...
        ,newResp+meanResponse,'poisson');
    [~,dev,~] = glmfit(ones(length(newResp),1),newResp+meanResponse,'poisson','constant','off');
    
    fprintf('\nExplained Deviance: %3.3f\n',1-minDev/dev);
    
    fprintf('AIC_Null: %3.3f  AIC_Model: %3.3f\n\n',2*1-dev,2*(1+transBases+histBases)-minDev);
    
    mov = loadimfile(celldata(ii).fullvalstimfile);
    
    prediction(ii).cellid = celldata(ii).cellid;
    prediction2(ii).cellid = celldata(ii).cellid;
    
    [~,~,trueFrames] = size(mov);
    corrs = zeros(trueFrames,1);
    for jj=2:trueFrames
        prevFrame = mov(:,:,jj-1);
        currentFrame = mov(:,:,jj);
        r = corrcoef(prevFrame(:),currentFrame(:));
        corrs(jj) = r(1,2);
    end
    transitionInds = 1-corrs;
    
    transDesign = zeros(trueFrames,numBack);
    temp = double(transitionInds);
    for jj=0:numBack-1
        forcorr = temp(1:end-jj);
        transDesign(:,jj+1) = forcorr;
        temp = [0;temp];
    end
    transDesign = transDesign(:,2:end);
    
    numIter = 5000;
    estResponse = zeros(trueFrames,numIter);
    for jj=1:numIter
        currentHist = poissrnd(exp(b(1)),[1,histLags]);
        for kk=1:trueFrames
            estResponse(kk,jj) = poissrnd(exp(b(1)+transDesign(kk,:)*...
                transbasisFuns*b(2:transBases+1)+...
                    currentHist*histbasisFuns*b(transBases+2:end)));
            currentHist = [estResponse(kk,jj),currentHist(1:end-1)];
        end
    end
    prediction(ii).response = median(estResponse,2);
    prediction2(ii).response = exp(b(1)+transDesign*...
                transbasisFuns*b(2:transBases+1));
end

save('Predictions_Transitions.mat','prediction');
save('Transitions.mat','rfs','prediction2');