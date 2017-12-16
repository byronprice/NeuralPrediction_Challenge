% runASD_STRF.m

cellinfo

numCells = length(celldata);

prediction = struct('cellid',cell(numCells,1),'response',zeros(713,1));
rfs = cell(numCells,4);
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
    
    numBack = 13;
    fullRF = zeros(DIM*DIM,numBack);
    data = zeros(numFrames,numBack);
    
    meansToSubtract = zeros(numBack,DIM*DIM);
    for kk=1:numBack
        count = 1;
        for jj=inds'
            temp = mov(:,:,max(jj-kk-1,1)); % try -1 ... back two frames
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
    
    [fullEst,~,~] = fastASD(data,newResp,numBack,0.1);
    
    figure;subplot(2,1,1);
    plot(max(data*fullEst+responseMean,0),newResp+responseMean,'.');
    title(sprintf('Cell %s',celldata(ii).cellid));
    xlabel('Predicted Spiking');
    ylabel('True Spiking');
    subplot(2,1,2);
    plot(1:numBack,fullEst);
    xlabel('Frame into Past');
    
    r = corrcoef(max(data*fullEst+responseMean,0),newResp);
    
    fprintf('\n\nCorrelation: %3.3f\n\n',r(1,2));
    pause(5);
    
    rfs{ii,1} = fullRF;
    rfs{ii,2} = fullEst;
    rfs{ii,3} = responseMean;
    rfs{ii,4} = meansToSubtract;
    
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
            
            if jj-kk-1 < 1
                temp2 = mov(20:50,20:50,:);
                temp = mean(temp2(:)).*ones(DIM1,DIM2);
            else
                temp = mov(:,:,max(jj-kk-1,1));
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
    prediction(ii).response = max(tempPreds*fullEst+responseMean,0);
end

save('ASD_PredictionsSTRF.mat','prediction');
save('ASD_STRFs.mat','rfs');