% runASD_FullSTRF.m

cellinfo

numCells = length(celldata);

prediction = struct('cellid',cell(numCells,1),'response',zeros(713,1));
rfs = cell(numCells,3);
for ii=1:numCells
    load(celldata(ii).datafile,'resp');
    inds = find(~isnan(resp));
    newResp = resp(~isnan(resp));
    numFrames = length(newResp);
    mov = loadimfile(celldata(ii).fullstimfile);
    
    meanResponse = mean(newResp);
    newResp = newResp-meanResponse;
    
    [DIM,~,~] = size(mov);
    
    numBack = 15;
    newMov = zeros(numFrames,DIM*DIM*numBack);
    count = 1;
    for jj=inds'
        miniMov = zeros(DIM,DIM,numBack);
        for kk=1:numBack
            temp = mov(:,:,max(jj-kk+1,1));
            miniMov(:,:,kk) = temp;
        end
        newMov(count,:) = miniMov(:)';
        count = count+1;
    end
    meanToSubtract = sum(newMov,1)./numFrames;
    newMov = newMov-meanToSubtract;
    [fullSTRF,~,~] = fastASD(newMov,newResp,[DIM,DIM,numBack],[2,2,5]);
    
    figure;plot(newResp+meanResponse,max(newMov*fullSTRF+meanResponse,0),'.');
    title(sprintf('Cell %s',celldata(ii).cellid));
    xlabel('True Spiking');
    ylabel('Predicted Spiking');
    
    r = corrcoef(max(newMov*fullSTRF+meanResponse,0),newResp);
    
    fprintf('\n\n\nCorrelation: %3.3f\n\n\n',r(1,2));
    
    rfs{ii,1} = fullSTRF;
    rfs{ii,2} = meanToSubtract;
    rfs{ii,3} = meanResponse;
    
    mov = loadimfile(celldata(ii).fullvalstimfile);
    
    prediction(ii).cellid = celldata(ii).cellid;
    
    [DIM1,DIM2,numFrames] = size(mov);
    
    if DIM1*DIM2 ~= DIM*DIM
        newMov = zeros(numFrames,DIM*DIM*numBack);
        miniMov = zeros(DIM,DIM,numBack);
    else
        newMov = zeros(numFrames,DIM1*DIM2*numBack);
        miniMov = zeros(DIM1,DIM2,numBack);
    end
     
    centralMean = mean(mean(mean(mov(20:50,20:50,:))));
    for jj=1:numFrames
        for kk=1:numBack
            if jj-kk-1 < 1
                temp = centralMean.*ones(DIM1,DIM2);
            else
                temp = mov(:,:,max(jj-kk+1,1));
            end
            
            if DIM1*DIM2 ~= DIM*DIM
                temp = imresize(temp,DIM/DIM1);
            end
            miniMov(:,:,kk) = temp;
        end
        newMov(jj,:) = temp(:)';
    end
    newMov = newMov-meanToSubtract;
    prediction(ii).response = max(newMov*fullSTRF+meanResponse,0);
end

save('ASD_PredictionsFullSTRF.mat','prediction');
save('ASD_FullSTRFs.mat','rfs');