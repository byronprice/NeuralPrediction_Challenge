% runASD_FullSTRF.m

cellinfo

numCells = length(celldata);

prediction = struct('cellid',cell(numCells,1),'response',zeros(713,1));
rfs = cell(numCells,4);

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
    
    [trueDim,~,movFrames] = size(mov);
    DIM = [40,40];
    
    tempMov = zeros(DIM(1),DIM(2),movFrames);
    for jj=1:movFrames
        tempIm = mov(:,:,jj);
        tempIm = imresize(tempIm,DIM(1)/trueDim);
        tempMov(:,:,jj) = tempIm;
    end
    clear mov;
    
    numBack = 25;
    newMov = zeros(numFrames,DIM(1)*DIM(2)*numBack);
    count = 1;
    for jj=inds'
        miniMov = zeros(DIM(1),DIM(2),numBack);
        for kk=1:numBack
            temp = tempMov(:,:,max(jj-kk-2,1));
            miniMov(:,:,kk) = temp;
        end
        newMov(count,:) = miniMov(:)';
        count = count+1;
    end
    meanToSubtract = sum(newMov,1)./numFrames;
    newMov = newMov-meanToSubtract;
    [fullSTRF,~,~] = fastASD(newMov,newResp,[DIM(1),DIM(2),numBack],[2,2,1]);
    
    numIter = 250;result = zeros(numIter,2);
    allInds = 1:numFrames;
    for jj=1:numIter
        inds = randperm(numFrames,round(numFrames.*0.75));
        inds = ismember(allInds,inds);
        holdOutInds = ~inds;
        
        [b,~,~] = glmfit(newMov(inds,:)*fullSTRF,newResp(inds),'poisson');
        r = corrcoef(exp(newMov(holdOutInds,:)*fullSTRF*b(2:end)+b(1)),newResp(holdOutInds));
        result(jj,1) = r(1,2);
        
        r = corrcoef(max(newMov(holdOutInds,:)*fullSTRF+meanResponse,0),newResp(holdOutInds));
        result(jj,2) = r(1,2);
    end
    
    result = median(result,1);
    display(result);
    
    if result(1)>result(2)
       poisson = true; 
       [b,~,~] = glmfit(newMov*fullSTRF,newResp,'poisson');
    else
       poisson = false; 
    end
    
    rfs{ii,1} = fullSTRF;
    rfs{ii,2} = meanToSubtract;
    rfs{ii,3} = meanResponse;
    rfs{ii,4} = poisson;
    
    mov = loadimfile(celldata(ii).fullvalstimfile);
    
    prediction(ii).cellid = celldata(ii).cellid;
    
    [DIM1,DIM2,numFrames] = size(mov);
    
    newMov = zeros(numFrames,DIM(1)*DIM(2)*numBack);
    miniMov = zeros(DIM(1),DIM(2),numBack);
     
    centralMean = mean(mean(mean(mov(20:50,20:50,:))));
    for jj=1:numFrames
        for kk=1:numBack
            if jj-kk-2 < 1
                temp = centralMean.*ones(DIM1,DIM2);
            else
                temp = mov(:,:,max(jj-kk-2,1));
            end
            
            temp = imresize(temp,DIM(1)/DIM1);
            miniMov(:,:,kk) = temp;
        end
        newMov(jj,:) = temp(:)';
    end
    newMov = newMov-meanToSubtract;
    
    if poisson == true
        prediction(ii).response = exp(newMov*fullSTRF*b(2)+b(1));
    else
        prediction(ii).response = max(newMov*fullSTRF+meanResponse,0);
    end
end

save('ASD_PredictionsFullSTRF.mat','prediction');
save('ASD_FullSTRFs.mat','rfs');