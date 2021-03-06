function [] = TestRF_CoorDesc()
% 3D Gabor wavelets with coordinate descent
warning('off','all');

train = 0.75;

cellinfo

numCells = length(celldata);

neuralResponse = cell(numCells,1);
for ii=1:numCells
    load(celldata(ii).datafile,'resp');
    neuralResponse{ii} = resp;
end

prediction = struct('cellid',cell(numCells,1),'response',zeros(713,1));

for ii=numCells:-1:1
    resp = neuralResponse{ii};
   
    load(sprintf('%s_data.mat',celldata(ii).cellid),'stim','vstim');
    
    numLags = 10;
    meanIm = mean(stim,3);
    tempStim = zeros(size(meanIm,1),size(meanIm,2),numLags);
    for jj=1:numLags
       tempStim(:,:,jj) = meanIm;
    end
    
    newStim = zeros(size(stim));
    for jj=1:size(stim,3)
        if jj==1
            newStim(:,:,jj) = stim(:,:,jj)-meanIm;
        else
            newStim(:,:,jj) = stim(:,:,jj)-meanIm; 
        end
    end
    stim = newStim;
    
    vstim = cat(3,tempStim,vstim);
    
    newStim = zeros(size(vstim));
    for jj=1:size(vstim,3)
        if jj==1
            newStim(:,:,jj) = vstim(:,:,jj)-meanIm;
        else
            newStim(:,:,jj) = vstim(:,:,jj)-meanIm; 
        end
    end
    vstim = newStim;
    
    allStims = cat(3,stim,vstim);
    
    prediction(ii).cellid = celldata(ii).cellid;
    
    params = preprocWavelets3d;
    params.dirdivisions = 12;
    params.fdivisions = 8; % 6 or 8, depending on phasemode
    params.veldivisions = 10; % 6
    params.tsize = numLags;
    params.phasemode = 3; % 0, 1, or 3
    params.zeromean_value = 0;
    % [origStim,params] = preprocWavelets3d(allStims, params);
    [origStim] = GenerateTSBases(allStims,7^5,[size(meanIm,1),size(meanIm,1),numLags]);
    data = resp;
    stim = origStim(1:length(data),:);
    
    nWts = size(stim,2);
    design = zeros(size(stim,1)-(numLags-1),nWts*numLags); % 10 delays
    
    for jj=1:numLags
        design(:,1+(jj-1)*nWts:nWts+(jj-1)*nWts) = stim(1+(jj-1):end-(numLags-1)+(jj-1),:);
    end
    data = data(numLags:end);
    inds = find(~isnan(data));
    data = data(inds);
    design = design(inds,:);
    
    bigMu = zeros(713,1);
    numDivisions = 5;
    divideInds = round(linspace(1,length(data),numDivisions+1));
    
    for mm=1:numDivisions
        testIdx = divideInds(mm):divideInds(mm+1);
        allInds = 1:length(data);
        temp = ~ismember(allInds,testIdx);
        trainIdx = find(temp);
        
        weights = zeros(nWts*numLags+1,1);
        link = @(x) exp(x);
        weights(1) = log(mean(data));
%         likelihood = @(y,mu) -sum((y-mu).^2);% negative normal log-likelihood
        likelihood = @(y,mu)  y.*log(y./mu)-(y-mu); % negative poisson deviance
        
        weightInds = 2:length(weights);
        
        % myCost = @(y,mu) y.*log(y./mu)-(y-mu); % poisson deviance
%         likelihood = @(y,mu) sum(-mu-log(factorial(y))+y.*log(mu)); % divide by factorial(y)
        %  if values can be greater than 1
        
        mu = link(weights(1)+design(trainIdx,:)*weights(weightInds));
        
        % cost = myCost(data(trainIdx),mu);
        % cost(isnan(cost)) = mu(isnan(cost));
        % prevCost = 2*sum(cost);
        
        prevLikely = likelihood(data(trainIdx),mu);
        prevLikely(isnan(prevLikely) | isinf(prevLikely)) = mu(isnan(prevLikely) | isinf(prevLikely));
        prevLikely = -2*sum(prevLikely);
        
        numIter = 5e4;
        numParams = length(weightInds);
        
        grad = zeros(numParams,1);
        dt = 0.0001;
        
        lineSearch = [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8];
        lineLen = length(lineSearch);
        lineCost = zeros(lineLen,1);
        
        % figure;
        numToKeep = 100;
        lastTwenty = zeros(numToKeep,1);
        mu = link(weights(1)+design(testIdx,:)*weights(weightInds));
        tmp = likelihood(data(testIdx),mu);
        tmp(isnan(tmp) | isinf(tmp)) = mu(isnan(tmp) | isinf(tmp));
        tmp = -2*sum(tmp);
        lastTwenty(1) = tmp;
        
        lastTwentyWeights = zeros(length(weights),numToKeep);
        lastTwentyWeights(:,1) = weights;
        
        tolerance = 1e-3;
        for jj=1:numIter
            tempweights = weights(weightInds);
            nonZeroInds = find(tempweights~=0);
            mu = link(weights(1)+design(trainIdx,nonZeroInds)*tempweights(nonZeroInds));
            grad(:) = design(trainIdx,:)'*(data(trainIdx)-mu);
            [~,ind] = max(abs(grad));
            
            nonZeroInds = unique([nonZeroInds;ind]);
            currentDiff = 1;
            tempweights = weights(weightInds);
            prevWeight = weights(1+ind);
            prevGrad = grad(ind);
            linePrev = prevLikely;
            while currentDiff>tolerance
                for kk=1:lineLen
                    tempweights(ind) = prevWeight+dt*sign(prevGrad)*lineSearch(kk);
                    mu = link(weights(1)+design(trainIdx,nonZeroInds)*tempweights(nonZeroInds));
                    
                    tmp = likelihood(data(trainIdx),mu);
                    tmp(isnan(tmp) | isinf(tmp)) = mu(isnan(tmp) | isinf(tmp));
                    tmp = -2*sum(tmp);
                    lineCost(kk) = tmp;
                end
                [maxLikely,searchInd] = max(lineCost);
                
                if isinf(maxLikely)
                    tempweights(ind) = prevWeight;
                    mu = link(weights(1)+design(trainIdx,nonZeroInds)*tempweights(nonZeroInds));
                    tmp = likelihood(data(trainIdx),mu);
                    tmp(isnan(tmp) | isinf(tmp)) = mu(isnan(tmp) | isinf(tmp));
                    tmp = -2*sum(tmp);
                    maxLikely = tmp;
                    break;
                end
                prevWeight = prevWeight+dt*sign(prevGrad)*lineSearch(searchInd);
                tempweights(ind) = prevWeight;
                mu = link(weights(1)+design(trainIdx,nonZeroInds)*tempweights(nonZeroInds));
                prevGrad = design(trainIdx,ind)'*(data(trainIdx)-mu);
                
                currentDiff = maxLikely-linePrev;
                linePrev = maxLikely;
            end
            
            if maxLikely>=prevLikely
                prevLikely = maxLikely;
                weights(weightInds) = tempweights;
            else
                weights(1+ind) = weights(1+ind)+normrnd(0,dt);
            end
            
            mu = link(weights(1)+design(testIdx,nonZeroInds)*weights(weightInds(nonZeroInds)));
            
            tmp = likelihood(data(testIdx),mu);
            tmp(isnan(tmp) | isinf(tmp)) = mu(isnan(tmp) | isinf(tmp));
            tmp = -2*sum(tmp);
            heldOutLikely = tmp;
            lastTwenty = [heldOutLikely;lastTwenty(1:end-1)];
            
            lastTwentyWeights = [weights,lastTwentyWeights(:,1:end-1)];
            
            [maxVal,index] = max(lastTwenty);
            if sum((lastTwenty(1:index)-maxVal)<0) > numToKeep/2 && jj>numToKeep
                break;
            end
            
%             subplot(2,1,1);plot(jj,prevLikely,'.');hold on;
%             subplot(2,1,2);plot(jj,heldOutLikely,'.');hold on;
%             pause(1/100);
        end
        
        [maxLikelihood,ind] = max(lastTwenty);
        ind
        weights = lastTwentyWeights(:,ind);
        
        save(sprintf('%s_rfinfo_DiffIm%d.mat',celldata(ii).cellid,mm),'params','weights','link','maxLikelihood');
        
        
        vstim = origStim(end-721:end,:);
        vdesign = zeros(size(vstim,1)-(numLags-1),nWts*numLags); % 10 delays
        
        for jj=1:numLags
            vdesign(:,1+(jj-1)*nWts:nWts+(jj-1)*nWts) = vstim(1+(jj-1):end-(numLags-1)+(jj-1),:);
        end
        
        mu = link(weights(1)+vdesign*weights(weightInds));
        mu(isnan(mu))= mean(data);
        
        bigMu = bigMu+mu./numDivisions;
    end
    maxResp = max(data)+2;
    bigMu = max(bigMu,0);
    bigMu = min(bigMu,maxResp);
    figure;plot(bigMu);pause(1/50);
    prediction(ii).response = bigMu;
    save(sprintf('3DWavs_CoorDescent_TSBases-Exp%d.mat',ii),'prediction');
end

save('3DWavs_CoorDescent_TSBases-Exp.mat','prediction');

end
