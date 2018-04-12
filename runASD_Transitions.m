cellinfo

numCells = length(celldata);

prediction = struct('cellid',cell(numCells,1),'response',zeros(713,1));
%rfs = cell(numCells,3);

neuralResponse = cell(numCells,1);
for ii=1:numCells
    load(celldata(ii).datafile,'resp');
    neuralResponse{ii} = resp;
end

lastknot = 30;
c_pt_times_all = [-5 0 2 4 6 8 10 20 30];
s = 0.5;  % Define Tension Parameter

numBasisSeq = length(c_pt_times_all);
numknots = numBasisSeq;

% Construct spline matrix
Sseq = zeros(lastknot,numknots);
num_c_pts = length(c_pt_times_all);     % number of control points in total

for i=1:c_pt_times_all(3)
    nearest_c_pt_index = max(find(c_pt_times_all<i));
    nearest_c_pt_time = c_pt_times_all(nearest_c_pt_index);
    next_c_pt_time = c_pt_times_all(nearest_c_pt_index+1);
    
    u = (i-nearest_c_pt_time)/(next_c_pt_time-nearest_c_pt_time);
    
    p=[u^3 u^2 u 1]*[-s -s s-2 s;2*s s 3-2*s -s;-s 0 s 0;0 0 0 0];
    Sseq(i,:) = [zeros(1,nearest_c_pt_index-2) p zeros(1,num_c_pts-4-(nearest_c_pt_index-2))];
    
end

for i=c_pt_times_all(3)+1:c_pt_times_all(end-1)
    nearest_c_pt_index = max(find(c_pt_times_all<i));
    nearest_c_pt_time = c_pt_times_all(nearest_c_pt_index);
    next_c_pt_time = c_pt_times_all(nearest_c_pt_index+1);
    
    u = (i-nearest_c_pt_time)/(next_c_pt_time-nearest_c_pt_time);
    
    p=[u^3 u^2 u 1]*[-s 2-s s-2 s;2*s s-3 3-2*s -s;-s 0 s 0;0 1 0 0];
    Sseq(i,:) = [zeros(1,nearest_c_pt_index-2) p zeros(1,num_c_pts-4-(nearest_c_pt_index-2))];
    
end

for i = c_pt_times_all(end-1)+1:c_pt_times_all(end)
    nearest_c_pt_index = max(find(c_pt_times_all<i));
    nearest_c_pt_time = c_pt_times_all(nearest_c_pt_index);
    next_c_pt_time = c_pt_times_all(nearest_c_pt_index+1);
    
    u = (i-nearest_c_pt_time)/(next_c_pt_time-nearest_c_pt_time);
    p=[u^3 u^2 u 1]*[-s 2-s s -s;2*s s-3 -2*s -s;-s 0 s 0;0 1 0 0];
    Sseq(i,:) = [zeros(1,nearest_c_pt_index-2) p(1:3)];
    
end
    
for ii=1:numCells
    resp = neuralResponse{ii};
    
    histBases = 30;
    histDesign = zeros(length(resp),histBases);
    temp = resp;
    for jj=1:histBases
       temp = [0;temp(1:end-1)];
       histDesign(:,jj) = temp;
    end
    
    inds = find(~isnan(sum(histDesign,2)) & ~isnan(resp));
    newResp = resp(inds);
    histDesign = histDesign(inds,:);
    numFrames = length(newResp);
    mov = loadimfile(celldata(ii).fullstimfile);
    
    DIM = size(mov,1);
    
    convertMov = zeros(numFrames,DIM*DIM);
    count = 1;
    for jj=inds'
        temp = mov(:,:,jj);
        convertMov(count,:) = temp(:);
        count = count+1;
    end
    
    meanResponse = mean(newResp);
    newResp = newResp-meanResponse;
    
    [~,~,trueFrames] = size(mov);
    corrs = zeros(trueFrames,2);corrs(1,:) = 0;
    for jj=2:trueFrames
        prevFrame = mov(:,:,jj-1);
        currentFrame = mov(:,:,jj);
        r = corrcoef(prevFrame(:),currentFrame(:));
        vals = prevFrame(:)-currentFrame(:);
        corrs(jj,1) = sum(vals(vals>0));
        corrs(jj,2) = sum(abs(vals(vals<0)));
    end
    corrs(:,1) = corrs(:,1)./max(corrs(:,1));
    corrs(:,2) = corrs(:,2)./max(corrs(:,2));
    transitionInds = corrs;
    
    numBack = 30;transDesign = zeros(trueFrames,numBack*2);
    count = 1;
    for kk=1:2
        corrPattern = zeros(numBack,1);
        temp = double(transitionInds(:,kk));
        for jj=0:numBack-1
            forcorr = temp(1:end-jj);
            transDesign(:,count) = forcorr;
            r = corrcoef(newResp,forcorr(inds));
            corrPattern(jj+1) = r(1,2);
            temp = [0;temp];
            count = count+1;
        end
    end
    
    transDesign = transDesign(inds,:);
    
    [b,dev2,~] = glmfit([histDesign,transDesign*[Sseq;Sseq]],newResp+meanResponse,'poisson');
    [~,dev1,~] = glmfit(histDesign,newResp+meanResponse,'poisson');
    1-dev2/dev1
    b = [b(1);b(histBases+2:end)];

    mov = loadimfile(celldata(ii).fullvalstimfile);
    
    prediction(ii).cellid = celldata(ii).cellid;
    
    [~,~,trueFrames] = size(mov);
    corrs = zeros(trueFrames,2);corrs(1,:) = 0;
    for jj=2:trueFrames
        prevFrame = mov(:,:,jj-1);
        currentFrame = mov(:,:,jj);
        r = corrcoef(prevFrame(:),currentFrame(:));
        vals = prevFrame(:)-currentFrame(:);
        corrs(jj,1) = sum(vals(vals>0));
        corrs(jj,2) = sum(abs(vals(vals<0)));
    end
    corrs(:,1) = corrs(:,1)./max(corrs(:,1));
    corrs(:,2) = corrs(:,2)./max(corrs(:,2));
    transitionInds = corrs;
    
    numBack = 30;transDesign = zeros(trueFrames,numBack*2);
    count = 1;
    for kk=1:2
        temp = double(transitionInds(:,kk));
        for jj=0:numBack-1
            forcorr = temp(1:end-jj);
            transDesign(:,count) = forcorr;
            temp = [0;temp];
            count = count+1;
        end
    end
    
    response = exp(b(1)+(transDesign*[Sseq;Sseq])*b(2:end));
%     figure;plot(response);
    prediction(ii).response = response;
end

save('ASD_Predictions_Transitions.mat','prediction');
% save('ASD_Transitions.mat','rfs');