function [photoPatches, sketchPatches]=LoadBothPatches(photosPath, sketchesPath)
% photoPatches: M x N matrix, where N is the number of patches. M is the
% dimension of a patch.
% same to sketchPatches.
    fileList=getAllFiles(photosPath);
    photoPatches=[];
    sketchPatches=[];
    
    for i=1:length(fileList)
         photoPath=fileList{i};
        [pathStr, fileName, fileExt]=fileparts(photoPath);
        
        if strcmp(fileExt, '.mat')~=1
            continue;
        end
        
        newFileName=fileName;
        sketchPath=[sketchesPath, '/', newFileName, '.mat'];
        
        photo=load(photoPath);
        sketch=load(sketchPath);
        photoPatches=[photoPatches; photo.HOG];
        sketchPatches=[sketchPatches; sketch.HOG];
    end
    photoPatches=photoPatches';
    sketchPatches=sketchPatches';
end