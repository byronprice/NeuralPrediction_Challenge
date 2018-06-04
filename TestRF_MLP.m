function [] = TestRF_MLP()
% multi-layer perceptron trained with backprop to learn output neural
%  activity
% 3D Gabor wavelets 
warning('off','all');

train = 0.8;

cellinfo

numCells = length(celldata);

neuralResponse = cell(numCells,1);
for ww=1:numCells
    load(celldata(ww).datafile,'resp');
    neuralResponse{ww} = resp;
end

prediction1 = struct('cellid',cell(numCells,1),'response',zeros(713,1));
prediction2 = struct('cellid',cell(numCells,1),'response',zeros(713,1));

for ii=1:numCells
    resp = neuralResponse{ii};
   
    load(sprintf('%s_data.mat',celldata(ii).cellid),'stim','vstim');
    
    numLags = 10;
    meanIm = mean(stim,3);
    tempStim = zeros(size(meanIm,1),size(meanIm,2),numLags);
    for jj=1:numLags
       tempStim(:,:,jj) = meanIm;
    end
    
    allStims = cat(3,stim,tempStim);
    allStims = cat(3,allStims,vstim);
    
    prediction1(ii).cellid = celldata(ii).cellid;
    prediction2(ii).cellid = celldata(ii).cellid;
    
    params = preprocWavelets3d;
    params.dirdivisions = 12;
    params.fdivisions = 6; % 6 or 8, depending on phasemode
    params.veldivisions = 6; % 6
    params.tsize = numLags;
    params.phasemode = 3; % 0, 1, or 3
    [origStim,params] = preprocWavelets3d(allStims, params);
    
    data = resp;
    stim = origStim(1:length(resp),:);
    
    nWts = size(stim,2);
    design = zeros(size(stim,1)-(numLags-1),nWts*numLags); % 10 delays
    
    for jj=1:numLags
        design(:,1+(jj-1)*nWts:nWts+(jj-1)*nWts) = stim(1+(jj-1):end-(numLags-1)+(jj-1),:);
    end
    data = data(numLags:end);
    inds = find(~isnan(data));
    data = data(inds);
    design = design(inds,:);
    
    keepInds = [];
    
    for mm=1:5
        filename = sprintf('%s_rfinfo%d.mat',celldata(ii).cellid,mm);
        load(filename,'weights');
        inds = find(weights);
        inds = inds(2:end);
        inds = inds-1;
        keepInds = [keepInds;inds];
    end
    
    [keepInds2,ia,~] = unique(keepInds);
    allVals = (1:length(keepInds))';
    nonUnique = unique(keepInds(~ismember(allVals,ia)));
%     nonUnique
    design = design(:,keepInds2);
    design = [design,ones(size(design,1),1)];
    
    vstim = origStim(end-721:end,:);
    vdesign = zeros(size(vstim,1)-(numLags-1),nWts*numLags); % 10 delays
    
    for jj=1:numLags
        vdesign(:,1+(jj-1)*nWts:nWts+(jj-1)*nWts) = vstim(1+(jj-1):end-(numLags-1)+(jj-1),:);
    end
    
    vdesign = vdesign(:,keepInds2);
    vdesign = [vdesign,ones(size(vdesign,1),1)];
    
    firstLayer = size(design,2);
    hiddenLayer = firstLayer*2;
    finalLayer = 1;
    
    numDivisions = 5;
    divideInds = round(linspace(1,length(data),numDivisions+1));
    
    % STOCHASTIC GRADIENT DESCENT
    batchSize = 10; % make mini batches and run the algorithm
    % on those "runs" times
    runs = 2e6;

    eta = 0.01;
    lambdaVals = [0,0.1,1,10,25,50,100,150,250,500,1000,1500,2000,5000];
    
    heldOutDev = zeros(numDivisions,length(lambdaVals));
    predictions = zeros(713,numDivisions,length(lambdaVals));
  %  figure(1);
    for mm=1:numDivisions
        testIdx = divideInds(mm):divideInds(mm+1);
        allInds = 1:length(data);
        temp = ~ismember(allInds,testIdx);
        trainIdx = find(temp);
        
        trainDesign = design(trainIdx,:);
        testDesign = design(testIdx,:);
        
        trainData = data(trainIdx);
        testData = data(testIdx);
        
        trainN = size(trainDesign,1);
        
        lambdacount = 1;
        for lambda = lambdaVals
            % initialize perceptron and run gradient descent algo
            myNet = Network([firstLayer,hiddenLayer,finalLayer]);
            alpha = 1; % proportion of hidden nodes to keep during dropout
            
            numCalcs = myNet.numCalcs;
            dCostdWeight = cell(1,numCalcs);
            dCostdBias = cell(1,numCalcs);
            %         figure;
            
            numToKeep = 2000;
            oldPreds = zeros(713,numToKeep);
            oldExpVars = zeros(1,numToKeep);
            for ww=1:runs
                indeces = ceil(rand([batchSize,1]).*trainN);
                [dropOutNet,dropOutInds] = MakeDropOutNet(myNet,alpha);
                for jj=1:numCalcs
                    layer1 = dropOutNet.layerStructure(jj);
                    layer2 = dropOutNet.layerStructure(jj+1);
                    dCostdWeight{jj} = zeros(layer1,layer2);
                    dCostdBias{jj} = zeros(layer2,1);
                end
                for jj=1:batchSize
                    index = indeces(jj);
                    
                    [costweight,costbias] = BackProp(trainDesign(index,:)',dropOutNet,...
                        trainData(index));
                    for kk=1:numCalcs
                        dCostdWeight{kk} = dCostdWeight{kk}+costweight{kk};
                        dCostdBias{kk} = dCostdBias{kk}+costbias{kk};
                    end
                end
                [dropOutNet] = GradientDescent(dropOutNet,dCostdWeight,dCostdBias,batchSize,eta,trainN,lambda);
                [myNet] = RevertToWholeNet(dropOutNet,myNet,dropOutInds);
                %     clear indeces;% dCostdWeight dCostdBias;
                
                tempNet = AdjustDropOutNet(myNet,alpha);
                predictTrain = zeros(length(testData),1);
                
                for kk=1:length(testData)
                    [~,Z] = Feedforward(testDesign(kk,:)',tempNet);
                    predictTrain(kk) = max(Z{end},0);% exp(Z{end});
                end
                
%                 % modelDev = GetDeviance(testData,predictTrain);
%                 initialDev = testData.*log(testData./predictTrain)-(testData-predictTrain);
%                 initialDev(isnan(initialDev) | isinf(initialDev)) = predictTrain(isnan(initialDev) | isinf(initialDev));
%                 modelDev = 2*sum(initialDev);
%                 
%                 nullEst = mean(testData).*ones(length(testData),1);
%                 % nullDev = GetDeviance(testData,nullEst);
%                 initialDev = testData.*log(testData./nullEst)-(testData-nullEst);
%                 initialDev(isnan(initialDev) | isinf(initialDev)) = predictTrain(isnan(initialDev) | isinf(initialDev));
%                 nullDev = 2*sum(initialDev);
%                 tempHeldOut = 1-modelDev/nullDev;

                nullVar = var(testData);
                modelVar = var(predictTrain-testData);
                heldOutExpVar = 1-modelVar/nullVar;
                
                oldExpVars = [heldOutExpVar,oldExpVars(1:end-1)];
                
                finalPred = zeros(size(vdesign,1),1);
                
                for kk=1:size(vdesign,1)
                    [~,Z] = Feedforward(vdesign(kk,:)',tempNet);
                    finalPred(kk) = max(Z{end},0);% exp(Z{end});
                end
                oldPreds = [finalPred,oldPreds(:,1:end-1)];
                
                [maxVal,myind] = max(oldExpVars);
                if sum((oldExpVars(1:myind)-maxVal)<0) >= numToKeep-1 && ww>=numToKeep
                    break;
                end
            end
            [maxExpVar,ind] = max(oldExpVars);
            predictions(:,mm,lambdacount) = oldPreds(:,ind);
            heldOutDev(mm,lambdacount) = maxExpVar;
            lambdacount = lambdacount+1;
        end
    end
    fprintf('\n\n');
    ii
    myDevs = mean(heldOutDev,1)-var(heldOutDev,[],1);
    myPreds = squeeze(mean(predictions,2));
    [~,ind] = max(myDevs);
    prediction1(ii).response = myPreds(:,ind);
    
    myDevs = mean(heldOutDev,1);
    [maxDev,ind] = max(myDevs);
    ind
    maxDev
    prediction2(ii).response = myPreds(:,ind);
end
prediction = prediction1;
save('3DWavs_MLP-Identity3.mat','prediction');
prediction = prediction2;
save('3DWavs_MLP-Identity4.mat','prediction');
end

