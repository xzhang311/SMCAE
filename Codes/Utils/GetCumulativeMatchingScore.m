function [accuracy]=GetCumulativeMatchingScore
imgFolder='/home/xi/Project/MCAE_FaceSketch/Data/hksketch/CUFSF/Testing/xi_Hogs_photos';
skhFolder='/home/xi/Project/MCAE_FaceSketch/Data/hksketch/CUFSF/Testing/xi_Hogs_sketches';

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

 imgSize=sqrt(size(imgFeatures, 2));
 [imgFeatures] = normalizeData(imgFeatures, imgSize, imgSize);
 [skhFeatures] = normalizeData(skhFeatures, imgSize, imgSize);

% img: rows
% skh: cols
dist=pdist2(skhFeatures, imgFeatures);
save('dist.mat', 'dist');
dist=pdist2(skhFeaturesOrig, imgFeatures);
save('distOrig.mat', 'dist');
dist=mat2gray(dist);

for i=1:size(dist, 1)
    r=dist(i, :);
    [sr, idx]=sortrows(r(:));
    sortDist(i, :)=sr';
    sortDistIdx(i, :)=idx';
end

% from top-ranking to 10-th ranking
for i=1:10
    retrived=sortDistIdx(:, 1:i);
    rowSize=size(retrived, 1);
    hitCount=0;
    for j=1:rowSize
        row=retrived(j, :);
        hitCount=hitCount+sum(row==j);
    end
    accuracy(i)=hitCount/rowSize;
end
accuracy
save('imgFeatures.mat', 'imgFeatures');
save('skhFeatures.mat', 'skhFeatures');
save('accuracy.mat', 'accuracy');
end
