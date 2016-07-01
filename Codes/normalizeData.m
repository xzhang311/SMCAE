function [outpatches] = normalizeData(patches, img_height, img_width)

% % Squash data to [0.1, 0.9] since we use sigmoid as the activation
% % function in the output layer
% 
% % Remove DC (mean of images). 
% patches1mean=mean(patches1);
% patches1 = bsxfun(@minus, patches1, patches1mean);
% patches2 = bsxfun(@minus, patches2, patches1mean);
% 
% % Truncate to +/-3 standard deviations and scale to -1 to 1
% patches1std=std(patches1(:));
% pstd = 3 * patches1std;
% patches1 = max(min(patches1, pstd), -pstd) / pstd;
% patches2 = max(min(patches2, pstd), -pstd) / pstd;
% 
% % Rescale from [-1,1] to [0.1,0.9]
% outpatches1 = (patches1 + 1) * 0.4 + 0.1;
% outpatches2 = (patches2 + 1) * 0.4 + 0.1;
    
%     outpatches=patches;
%     imgnum=size(patches, 2);
%     
%     for i=1:imgnum
%         img=reshape(patches(:, i), img_height, img_width);
%         
%         min_v=min(min(img));
%         max_v=max(max(img));
%         
%         img=(img-min_v)/(max_v-min_v);
%         outpatches(:, i)=max(0, img(:)-0.2)+0.1;
%     end
    
    min_v = min(patches')';
    max_v = max(patches')';
    
    min_v_mat = repmat ( min_v, 1, size(patches, 2) );
    max_v_mat = repmat ( max_v, 1, size(patches, 2) );
    
    outpatches = (patches - min_v_mat) ./ (max_v_mat - min_v_mat);
    outpatches(isnan(outpatches)==1)=1;
end
