% runDCT_STRF.m

cellinfo

numCells = length(celldata);

prediction = struct('cellid',cell(numCells,1),'response',zeros(713,1));
rfs = cell(numCells,5);

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

    meanIm = mean(mov,3);
    for jj=1:numFrames
    	mov(:,:,jj) = mov(:,:,jj) - meanIm;
    end
    
    responseMean = mean(newResp);
    newResp = newResp-mean(newResp);
    
    [DIM,~,~] = size(mov);
    numBack = 30;

    dctDim = [round(3*DIM/4),round(3*DIM/4),round(3*numBack/4)];
    dctMov = zeros(numFrames,prod(dctDIM));
    fullDctDim = prod(dctDIM);
    
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
        R = R(1:dctDIM(1),1:dctDIM(2),1:dctDIM(3));
        dctMov(count,:) = R(:);
    end

    temp = sum(abs(dctMov),1);
    [~,ind] = sort(temp,'descend');
    coeffs = 1;
    fullNorm = sum(temp);
    
    

    numIter = 100;
    lPath = [5,4,3,2,1.5,1,0.75,0.5,0.25,0.1,0.05,0.02,0.01];
    holdOutCorr = zeros(length(lPath)+1,numIter);
    train = round(numFrames.*0.75);
    for zz=1:numIter
        inds = randperm(numFrames,train);
        inds = ismember(allInds,inds);
        holdOutInds = ~inds;
        
        Sdct = cov(reduceDctData(inds,:));
        [XP,~,~,~,~,~] = QUIC('path',Sdct,50,lPath, 1e-8, 2, 100);
        
        newXP = zeros(q,q,length(lPath)+1);
        newXP(:,:,1:length(lPath)) = XP;
        newXP(:,:,end) = (train-1).*pinv(reduceDctData'*reduceDctData);
        XP = newXP;clear newXP;
        
        for yy=1:length(lPath)+1
            X = XP(:,:,yy);
            estFilter = (1./(train-1)).*(X*reduceDctData(inds,:)'*newResp(inds));
            r = corrcoef(max(reduceDctData(holdOutInds,:)*estFilter+responseMean,0),newResp(holdOutInds));
            holdOutCorr(yy,zz) = r(1,2);
        end
    end
    
   holdOutCorr = median(holdOutCorr,2);
   [~,ind] = max(holdOutCorr);

   if ind == length(lPath)+1
   	  L = 0;
   else
      L = 50*lPath(ind);
   end

   Sdct = cov(reduceDctData);
   [X,~,~,~,~,~] = QUIC('default',Sdct,L,1e-8,2,200);

   fullEst = (1./(numFrames-1)).*(X*reduceDctData*newResp);
    
    r = corrcoef(max(reduceDctData*fullEst+responseMean,0),newResp);
    
    fprintf('\nCorrelation: %3.3f\n\n\n',r(1,2));
    pause(1);
    
    rfs{ii,1} = fullRF;
    rfs{ii,2} = fullEst;
    rfs{ii,3} = responseMean;
    rfs{ii,4} = meansToSubtract;
    rfs{ii,5} = r(1,2);
    
    mov = loadimfile(celldata(ii).fullvalstimfile);
    
    prediction(ii).cellid = celldata(ii).cellid;
    
    [DIM1,DIM2,numFrames] = size(mov);
    
    if DIM1*DIM2 ~= DIM*DIM
        newMov = zeros(numFrames,DIM*DIM);
    else
        newMov = zeros(numFrames,DIM1*DIM2);
    end
    
    tempPreds = zeros(numFrames,numBack);
    for kk=1:numBack
        for jj=1:numFrames
            
            if jj-kk-2 < 1
                temp2 = mov(20:50,20:50,:);
                temp = mean(temp2(:)).*ones(DIM1,DIM2);
            else
                temp = mov(:,:,max(jj-kk-2,1));
            end
            
            if DIM1*DIM2 ~= DIM*DIM
                temp = imresize(temp,DIM/DIM1);
            end
            newMov(jj,:) = temp(:);
        end
        newMov = newMov - meansToSubtract(kk,:);
        newMov = newMov./10;
        tempPreds(:,kk) = newMov*fullRF(:,kk);
    end
    prediction(ii).response = exp(tempPreds*fullEst+b(1));
end

save('ASD_PredictionsSTRF_DCT.mat','prediction');
save('ASD_STRFs_DCT.mat','rfs');