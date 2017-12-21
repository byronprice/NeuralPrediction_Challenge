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
    
    numBack = 30; %Design = zeros(trueFrames,numBack);
    corrPattern = zeros(numBack,1);
    temp = double(transitionInds);
    for jj=0:numBack-1
        forcorr = temp(1:end-jj);
        r = corrcoef(newResp,forcorr(inds));
        corrPattern(jj+1) = r(1,2);
        temp = [0;temp];
    end
%     figure;plot(corrPattern);
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
    response = conv(transitionInds,corrPattern);
    response = response(1:trueFrames);
%     figure;plot(response);
    prediction(ii).response = max(response,0);
end

save('ASD_Predictions_Transitions.mat','prediction');
% save('ASD_Transitions.mat','rfs');