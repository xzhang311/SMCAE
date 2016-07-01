function [ out_patches ] = smoothImg( patches, img_height, img_width )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    out_patches=double(patches);

    for i=1:size(patches, 2)
        img=patches(:, i);
        img=double(reshape(img, img_height, img_width));
        g = fspecial('gaussian',[3, 3], 1);
        img = imfilter(img,g,'replicate');
        out_patches(:, i)=img(:);
    end

end

