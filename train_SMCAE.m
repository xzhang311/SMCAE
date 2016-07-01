function train_SMCAE(Beta, BalanceParam, outStr, imgFolder, sktFolder)
% Beta: weight for sparsity penalty.
% BalanceParam: weight for balance term
% outStr: output path for saving trained model of SMACE

addpath('./Codes');
addpath('./Codes/minFunc');
addpath('./Codes/Utils');

imgFolder='./Data/training/Hogs_photos';
sktFolder='./Data/training/Hogs_sketches';

imageFolder = imgFolder;
sketchFolder = sktFolder;
outputStr=outStr;

[actPatches, synPatches]=LoadBothPatches(imageFolder, sketchFolder);
       
imgSize=sqrt(size(actPatches, 1));

[actPatches] = normalizeData(actPatches, imgSize, imgSize);
[synPatches] = normalizeData(synPatches, imgSize, imgSize);

% rn=randi(size(synPatches,2),30,1);
% figure; display_network(actPatches(:, rn), imgSize);
% figure; display_network(synPatches(:, rn), imgSize);

%% 
%   Configuration of nodes num in each layer
%
%   The smallest number in this array represent the number of nodes in 
%   middlest layer. 
%
%   Three numbers in a row are: input layer, hidden layer, output layer.
%   One can easily expand the structure of the network by adding more
%   latyers.
layerNodes=[imgSize*imgSize,  1000,    imgSize*imgSize; % layer 1
            1000,      500,     1000;                   % layer 2
            500,      1000,    imgSize*imgSize];        % layer 3
        
nHiddenLayers = size(layerNodes, 1);

%% 
%   Setup configuration of layers.
% 
%   For encoding layers whose input layers is bigger than hidden layer, an
%   identical layer of input layer is used as the output layer.
%
%   For decoding layers whose input layers is smaller than hidden layer, use
%   original image data as output.
for i=1:nHiddenLayers
    SMCAE(i).visibleSize = layerNodes(i, 1);
    SMCAE(i).hiddenSize = layerNodes(i, 2);
    SMCAE(i).outputSize = layerNodes(i, 3);
    SMCAE(i).sparsityParam = 0.1;
    SMCAE(i).balanceParam = BalanceParam;
    SMCAE(i).lambda = 3e-3;
    SMCAE(i).beta = Beta;
    SMCAE(i).inActData = 0;
    SMCAE(i).inSynData = 0;
    SMCAE(i).outActData = 0;
    SMCAE(i).outSynData = 0;
    SMCAE(i).predictActData = 0;
    SMCAE(i).predictSynData = 0;
    SMCAE(i).theta = initializeParametersNonsymetric(SMCAE(i).hiddenSize,...
                                                     SMCAE(i).visibleSize,...
                                                     SMCAE(i).outputSize);
end

%%
% Train AE using synthetic data
%
SMCAE(1).inActData = actPatches;
SMCAE(1).inSynData = synPatches;
SMCAE(1).outActData = actPatches;
SMCAE(1).outSynData = actPatches;

%% 
%   Train stacked autoencoder
%   Layerwise training
options.Method = 'cg';
options.maxIter = 200; %200
options.display = 'on';

for i=1:nHiddenLayers
    % Train
    for j=1:20 %20
        i
        j
        [SMCAE(i).theta, SMCAE(i).cost] = minFunc( @(p) sparseAutoencoderCost_balanced_nonsymetric(p, ...
                                   SMCAE(i).visibleSize, SMCAE(i).hiddenSize, SMCAE(i).outputSize, ...
                                   SMCAE(i).lambda, SMCAE(i).sparsityParam, SMCAE(i).balanceParam, SMCAE(i).beta, ...
                                   SMCAE(i).inSynData, SMCAE(i).inActData, ...
                                   SMCAE(i).outSynData, SMCAE(i).outActData), ...
                                   SMCAE(i).theta, options);
        w1Len = SMCAE(i).hiddenSize * SMCAE(i).visibleSize;
        w2Len = SMCAE(i).hiddenSize * SMCAE(i).outputSize;
        b1Len = SMCAE(i).hiddenSize;
        b2Len = SMCAE(i).outputSize;
        
        SMCAE(i).W1 = reshape(SMCAE(i).theta(1 : w1Len), SMCAE(i).hiddenSize, SMCAE(i).visibleSize);
        SMCAE(i).lW2 = reshape(SMCAE(i).theta(w1Len+1 : w1Len+w2Len), SMCAE(i).outputSize, SMCAE(i).hiddenSize);
        SMCAE(i).rW2 = reshape(SMCAE(i).theta(w1Len+w2Len+1 : w1Len+2*w2Len), SMCAE(i).outputSize, SMCAE(i).hiddenSize);
        SMCAE(i).b1 = SMCAE(i).theta(w1Len+2*w2Len+1 : w1Len+2*w2Len+b1Len);
        SMCAE(i).lb2 = SMCAE(i).theta(w1Len+2*w2Len+b1Len+1 : w1Len+2*w2Len+b1Len+b2Len);
        SMCAE(i).rb2 = SMCAE(i).theta(w1Len+2*w2Len+b1Len+b2Len+1 : w1Len+2*w2Len+b1Len+2*b2Len);
    end
    
    % Predict output after training
    [SMCAE(i).predictSynData, SMCAE(i).predictActData]=sparseAutoencoderPredict_balanced_nonsymetric(SMCAE(i).theta, SMCAE(i).visibleSize, SMCAE(i).hiddenSize, SMCAE(i).outputSize,...
                                                              SMCAE(i).lambda, SMCAE(i).sparsityParam, SMCAE(i).balanceParam, SMCAE(i).beta,...
                                                              SMCAE(i).inSynData, SMCAE(i).inActData);
    % Initialize input/output of next layer.
    if i+1 < nHiddenLayers
        SMCAE(i+1).inActData=SMCAE(i).predictActData;
        SMCAE(i+1).inSynData=SMCAE(i).predictSynData;
        SMCAE(i+1).outActData=SMCAE(i+1).inActData;
        SMCAE(i+1).outSynData=SMCAE(i+1).inActData;
    end
    
    if i+1 == nHiddenLayers
        SMCAE(i+1).inActData=SMCAE(i).predictActData;
        SMCAE(i+1).inSynData=SMCAE(i).predictSynData;
        SMCAE(i+1).outActData=SMCAE(1).inActData;
        SMCAE(i+1).outSynData=SMCAE(1).inActData;
    end
    
    if i == nHiddenLayers
        [ndims, lm] = size(SMCAE(i).predictSynData);
        [ndims, rm] = size(SMCAE(i).predictActData);
        SMCAE(i).predictSynData = sigmoid(SMCAE(i).lW2 * SMCAE(i).predictSynData + repmat(SMCAE(i).lb2, 1, lm));
        SMCAE(i).predictActData = sigmoid(SMCAE(i).rW2 * SMCAE(i).predictActData + repmat(SMCAE(i).rb2, 1, rm));
    end
    
    save([outputStr, '/SMCAE.mat'], 'SMCAE');
end

%% 
%   Fine-tuning Stacked AE as a whole
    SMCAE_Theta = initializeSMCAEParameters(SMCAE);
    
    options.Method = 'cg'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
    options.maxIter = 200;	  % Maximum number of iterations of L-BFGS to run 
    options.display = 'on';

    for i=1:5
        [theta, cost]=minFunc(@(p) SMCAE_Cost(p, SMCAE), SMCAE_Theta, options);
        SMCAE_Theta = theta;
    end
%% 
%   update theta of SMCAE
synLastLayer=synPatches;
actLastLayer=actPatches;
synFeatures=[];
actFeatures=[];
thetaIdx = 1;
for i = 1: nHiddenLayers
    synInput=synLastLayer;
    actInput=actLastLayer;
    
    wLen = length(SMCAE(i).W1(:));
    SMCAE(i).W1 = reshape(SMCAE_Theta(thetaIdx : thetaIdx+wLen-1), SMCAE(i).hiddenSize, SMCAE(i).visibleSize); 
    thetaIdx = thetaIdx+wLen;
    bLen = length(SMCAE(i).b1(:));
    SMCAE(i).b1 = SMCAE_Theta(thetaIdx : thetaIdx+bLen-1);
    thetaIdx = thetaIdx+bLen;
    
    [ndims, lm] = size(synInput);
    [ndims, rm] = size(actInput);
    
    lz = SMCAE(i).W1 * synInput + repmat(SMCAE(i).b1, 1, lm);
    la = sigmoid(lz);
    rz = SMCAE(i).W1 * actInput + repmat(SMCAE(i).b1, 1, rm);
    ra = sigmoid(rz);
    
    SMCAE(i).predictSynData=la;
    SMCAE(i).predictActData=ra;
    
    synLastLayer=la;
    actLastLayer=ra;
    
    if i==nHiddenLayers
        synInput=synLastLayer;
        actInput=actLastLayer;
        lw2Len = length(SMCAE(i).lW2(:));
        SMCAE(i).lW2 = reshape(SMCAE_Theta(thetaIdx : thetaIdx+lw2Len-1), SMCAE(i).outputSize, SMCAE(i).hiddenSize);
        thetaIdx = thetaIdx+lw2Len;
        rw2Len = length(SMCAE(i).rW2(:));
        SMCAE(i).rW2 = reshape(SMCAE_Theta(thetaIdx : thetaIdx+rw2Len-1), SMCAE(i).outputSize, SMCAE(i).hiddenSize);
        thetaIdx = thetaIdx+rw2Len;
        lb2Len = length(SMCAE(i).lb2(:));
        SMCAE(i).lb2 = SMCAE_Theta(thetaIdx : thetaIdx+lb2Len-1);
        thetaIdx = thetaIdx+lb2Len;
        rb2Len = length(SMCAE(i).rb2(:));
        SMCAE(i).rb2 = SMCAE_Theta(thetaIdx : thetaIdx+rb2Len-1);
        
        lz = SMCAE(i).lW2 * synInput + repmat(SMCAE(i).lb2, 1, lm);
        la = sigmoid(lz);
        rz = SMCAE(i).rW2 * actInput + repmat(SMCAE(i).rb2, 1, rm);
        ra = sigmoid(rz);
        SMCAE(i).predictSynData=la;
        SMCAE(i).predictActData=ra;
    end
end

save([outputStr, '/SMCAE_FineTuned.mat'], 'SMCAE');   
    
end
    
    
    
    
    
    
    
    
