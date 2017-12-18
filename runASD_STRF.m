% runASD_STRF.m

cellinfo

numCells = length(celldata);

prediction = struct('cellid',cell(numCells,1),'response',zeros(713,1));
rfs = cell(numCells,5);
for ii=1:numCells
    load(celldata(ii).datafile,'resp');
    inds = find(~isnan(resp));
    newResp = resp(~isnan(resp));
    numFrames = length(newResp);
    mov = loadimfile(celldata(ii).fullstimfile);
    
    responseMean = mean(newResp);
    newResp = newResp-mean(newResp);
    
    [DIM,~,~] = size(mov);
    newMov = zeros(numFrames,DIM*DIM);
    
    numBack = 20;
    fullRF = zeros(DIM*DIM,numBack);
    data = zeros(numFrames,numBack);
    
    meansToSubtract = zeros(numBack,DIM*DIM);
    for kk=1:numBack
        count = 1;
        for jj=inds'
            temp = mov(:,:,max(jj-kk+1,1)); % try -1 ... back two frames
            newMov(count,:) = temp(:);
            count = count+1;
        end
        meansToSubtract(kk,:) = sum(newMov,1)./numFrames;
        newMov = newMov - sum(newMov,1)./numFrames;
        newMov = newMov./10;
        [kest,~,~] = fastASD(newMov,newResp,[DIM,DIM],1);
        fullRF(:,kk) = kest;
        data(:,kk) = newMov*kest;
    end
    
%     [fullEst,~,~] = fastASD(data,newResp,numBack,0.1);
    newResp = newResp+responseMean;
    numBases = 6;basisFuns = zeros(numBack,numBases);
    stdev = [0.1,0.5,1,2,3,4,5,6,7,8,9,10];
    numIter = 100;
    holdOutCorr = zeros(length(stdev),numIter);
    lags = 0:numBack-1;centerPoints = linspace(0,numBack-1,numBases);
    allInds = 1:numFrames;
    for zz=1:length(stdev)
        for yy=1:numBases
            basisFuns(:,yy) = exp(-(lags-centerPoints(yy)).^2./(2*stdev(zz)*stdev(zz)));
        end
        
        for xx=1:numIter
            inds = randperm(numFrames,round(numFrames.*0.7));
            inds = ismember(allInds,inds);
            holdOutInds = ~inds;
            
            [b,~,~] = glmfit(data(inds,:)*basisFuns,newResp(inds),'poisson');
            r = corrcoef(exp(data(holdOutInds,:)*basisFuns*b(2:end)+b(1)),newResp(holdOutInds));
            holdOutCorr(zz,xx) = r(1,2);
        end
    end
    holdOutCorr = median(holdOutCorr,2);
    [maxCorr,ind] = max(holdOutCorr);
    fprintf('\n\nMax Hold-Out Correlation: %3.3f\n',maxCorr);
    stdev = stdev(ind);
    for yy=1:numBases
        basisFuns(:,yy) = exp(-(lags-centerPoints(yy)).^2./(2*stdev*stdev));
    end
    b = glmfit(data*basisFuns,newResp,'poisson');
    fullEst = basisFuns*b(2:end);
    
    figure;subplot(2,1,1);
    plot(exp(data*fullEst+b(1)),newResp,'.');
    title(sprintf('Cell %s',celldata(ii).cellid));
    xlabel('Predicted Spiking');
    ylabel('True Spiking');
    subplot(2,1,2);
    plot(1:numBack,fullEst);
    xlabel('Frame into Past');
    
    r = corrcoef(exp(data*fullEst+b(1)),newResp);
    
    fprintf('\nCorrelation: %3.3f\n\n\n',r(1,2));
    pause(5);
    
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
            
            if jj-kk+1 < 1
                temp2 = mov(20:50,20:50,:);
                temp = mean(temp2(:)).*ones(DIM1,DIM2);
            else
                temp = mov(:,:,max(jj-kk+1,1));
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

save('ASD_PredictionsSTRF.mat','prediction');
save('ASD_STRFs.mat','rfs');