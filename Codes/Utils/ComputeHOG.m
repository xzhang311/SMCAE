function hogs=ComputeHOG(path)
    fileList=getAllFiles(path);
    
    hogs=zeros(length(fileList), 2268);
    
    for i=1:length(fileList)
        photoPath=fileList{i};
        photo=imread(photoPath);
        
        if ndims(photo)==3
            photo=rgb2gray(photo);
        end
        
        sz=24;
        [hog, visualization] = extractHOGFeatures(photo,'CellSize',[sz sz]);
        hogs(i, :)=hog(:);
    end
end