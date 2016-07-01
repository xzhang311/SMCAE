function [synFeatures, actFeatures, synError, actError]=predict_SMCAE(imgFolder, skhFolder, SMCAE)
    photoPatches=[];
    sketchPatches=[];
    
    fileList=getAllFiles(imgFolder);
    for i=1:length(fileList)
        imgPath=fileList{i};
        img=load(imgPath);
        photoPatches=[photoPatches; img.HOG];
    end
    
    fileList=getAllFiles(skhFolder);
    for i=1:length(fileList)
        skhPath=fileList{i};
        skh=load(skhPath);
        sketchPatches=[sketchPatches; skh.HOG];
    end
    
    imgSize=sqrt(size(photoPatches, 1));
    [actPatches] = normalizeData(photoPatches, imgSize, imgSize);
    [synPatches] = normalizeData(sketchPatches, imgSize, imgSize);
    
    actPatches=photoPatches;
    synPatches=sketchPatches;
    
    nLayers=length(SMCAE);
    synLastLayer=synPatches';
    actLastLayer=actPatches';
    synFeatures=[];
    actFeatures=[];
    
    for i=1:nLayers
        synInput=synLastLayer;
        actInput=actLastLayer;
        
        [ndims, lm] = size(synInput);
        [ndims, rm] = size(actInput);
        
        lz = SMCAE(i).W1 * synInput + repmat(SMCAE(i).b1, 1, lm);
        la = sigmoid(lz);
        rz = SMCAE(i).W1 * actInput + repmat(SMCAE(i).b1, 1, rm);
        ra = sigmoid(rz);
        
        synLastLayer=la;
        actLastLayer=ra;
        
        synFeatures=[synFeatures; la];
        actFeatures=[actFeatures; ra];
        
        if i == nLayers
            synInput=synLastLayer;
            actInput=actLastLayer;
            lz = SMCAE(i).lW2 * synInput + repmat(SMCAE(i).lb2, 1, lm);
            la = sigmoid(lz);
            rz = SMCAE(i).rW2 * actInput + repmat(SMCAE(i).rb2, 1, rm);
            ra = sigmoid(rz);
            synFeatures = [synFeatures; la];
            actFeatures = [actFeatures; ra];
            
            synError = sqrt(sum((la - synPatches').^2));
            actError = sqrt(sum((ra - actPatches').^2));
        end
    end
    synFeatures=synFeatures';
    actFeatures=actFeatures';
end
