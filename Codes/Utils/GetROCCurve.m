imgFolder='/home/xi/Project/MCAE_FaceSketch/Data/hksketch/CUFSF/Testing/EXTRACTED_FEATURES/HOG/photos';
skhFolder='/home/xi/Project/MCAE_FaceSketch/Data/hksketch/CUFSF/Testing/EXTRACTED_FEATURES/HOG/sketches';

fileList=getAllFiles(imgFolder);

for i=1:length(fileList)
    photoPath=fileList{i};
    photo=load(photoPath);
    f=photo.HOG(:)';
    imgFeatures(i, :)=f(:)';
end

fileList=getAllFiles(skhFolder);

for i=1:length(fileList);
    sketchPath=fileList{i};
    sketch=load(sketchPath);
    f=sketch.HOG(:)';
    skhFeatures(i, :)=f(:)';    
end

% img: rows
% skh: cols
dist=pdist2(skhFeatures, imgFeatures);
dist=mat2gray(dist);

sRate=0.001;
count=1;
for r = sRate:0.001:1
    retrievedMat=(dist<=r);
    totalPositive = size(dist, 1);
    TP = trace(retrievedMat);
    TPR = TP/totalPositive;
    
    FP = sum(sum(retrievedMat-diag(diag(retrievedMat))));
    totalNegative = size(dist, 1) * size(dist, 1) - size(dist, 1);
    FPR = FP/totalNegative;
    roc(count, :)=[FPR, TPR];
    count=count+1;
end

figure;
plot(roc(:, 1), roc(:, 2), 'k');
set(gca,'xscale','log') 
grid on;
hold on;



