function GetPatchesForImages(croppedImgPath, outPath)
croppedImgPath = '/home/xi/Project/MCAE_FaceSketch/Data/hksketch/CUFS_Student/Training/training_cropped_sketches';
outPath =  '/home/xi/Project/MCAE_FaceSketch/Data/hksketch/CUFS_Student/Training/xi_scaled_sketches';

    fileList=getAllFiles(croppedImgPath);
    
    for i=1:length(fileList)
        photoPath=fileList{i};
        [pathStr, fileName, fileExt]=fileparts(photoPath);
        newFileName=[fileName(1), fileName(3:end-4)];
        outPatchPath=[outPath, '/', newFileName, '.mat'];
        
        patchesArray = imread(photoPath);
        patchesArray = imresize(patchesArray, [50, 50]);
        
        if ndims(patchesArray)==3
            tmp=rgb2gray(patchesArray);
            patchesArray=tmp;
        end
        patchesArray=imadjust(patchesArray);
        patchesArray=mat2gray(patchesArray);
        
        patchesArray=patchesArray(:)';
        
        save(outPatchPath, 'patchesArray');
    end
    
end