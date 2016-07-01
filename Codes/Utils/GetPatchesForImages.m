function GetPatchesForImages(fullImgPath, croppedImgPath, ptPath, outPath)
% given faducial on full size image, this function first find a mapping
% between full image and cropped image. Then mapping faducial to cropped
% image.
% For each faducial on cropped image, this function crop a small patch
% centerred at a given faducial and save the total patches as a N*M matrix,
% where N is the number of faducials and M is the total size of pixels in
% small patches.
% 
fullImgPath = '/home/xi/Project/MCAE_FaceSketch/Data/hksketch/CUFS_Student/Testing/testing_photos';
croppedImgPath = '/home/xi/Project/MCAE_FaceSketch/Data/hksketch/CUFS_Student/Testing/testing_cropped_photos';
ptPath = '/home/xi/Project/MCAE_FaceSketch/Data/hksketch/CUFS_Student/Testing/testing_faducial_points_photos';
outPath =  '/home/xi/Project/MCAE_FaceSketch/Data/hksketch/CUFS_Student/Testing/xi_patches_photos';

    patchW=31;
    patchH=31;

    fileList=getAllFiles(fullImgPath);
    
    for i=1:length(fileList)
        photoPath=fileList{i};
        [pathStr, fileName, fileExt]=fileparts(photoPath);
        cropphotoPath=[croppedImgPath, '/', fileName, '.jpg'];
        pointPath=[ptPath, '/', fileName, '.dat'];
        outPatchPath=[outPath, '/', fileName, '.mat'];
        
        % load data
        photo=imread(photoPath);
        cropphoto=imread(cropphotoPath);
        pts=load(pointPath);
        
        % find homography mapping full image to cropped image
        % and update faducials.
        H=GetHomography(cropphoto, photo);
        pts=[pts, ones(size(pts, 1), 1)];
        newpts=inv(H)*pts';
        newpts=newpts';
        newpts(:, 1)=newpts(:, 1)./newpts(:, 3);
        newpts(:, 2)=newpts(:, 2)./newpts(:, 3);
        newpts(:, 3)=newpts(:, 3)./newpts(:, 3);
        pts=round(newpts);
        
        % change photo from rgb to gray scale.
        if ndims(cropphoto)==3
            cropphoto=rgb2gray(cropphoto);
        end
        cropphoto=imadjust(cropphoto);
        cropphoto=mat2gray(cropphoto); % value in [0, 1]
        
        % get patches
        patchesArray=zeros(size(pts, 1), patchW*patchH);
        
        % add margin to img to avoid out-boundary patches.
        newPhotoMargin=max(patchW, patchH)*2;
        newPhoto=ones(size(cropphoto, 1)+newPhotoMargin*2, size(cropphoto, 2)+newPhotoMargin*2);
        newPhoto(newPhotoMargin+1:newPhotoMargin+size(cropphoto, 1), newPhotoMargin+1:newPhotoMargin+size(cropphoto, 2))=cropphoto;
        pts(:, 1)=pts(:, 1)+newPhotoMargin;
        pts(:, 2)=pts(:, 2)+newPhotoMargin;
        
        for j=1:size(pts, 1)
            margin=floor(patchW/2);
            sy=pts(j, 1)-margin;
            ey=pts(j, 1)+margin;
            sx=pts(j, 2)-margin;
            ex=pts(j, 2)+margin;
            ptch=newPhoto(sy:ey, sx:ex);
            patchesArray(j, :)=ptch(:);
        end
        
        % save patches
        save(outPatchPath, 'patchesArray');
    end
    
    
end