real_path = '/home/xi/Project/LearningSyntheticRoof/data/Mine/Training/Flipped_Noised/Real/gable/real_400.mat';
tmplt_path = '/home/xi/Project/LearningSyntheticRoof/data/Mine/Training/Flipped_Noised/Template/gable/tmplt_400.mat';

real_patches=load(real_path);
tmplt_patches=load(tmplt_path);

imgnum=size(real_patches.real_400, 1);

out_real_patches=zeros(64*64, imgnum);
out_tmplt_patches=zeros(64*64, imgnum);

for i=1:imgnum
    img=reshape(real_patches.real_400(i, :, :), 128, 256);
    img=im2bw(imresize(img, [64, 64], 'bicubic'), 1.0/125.0);
    out_real_patches(:, i)=img(:);
    
    img=reshape(tmplt_patches.tmplt_400(i, :, :), 128, 256);
    img=imresize(img, [64, 64], 'bicubic');
    out_tmplt_patches(:, i)=img(:);
end

rn=randi([1, imgnum], 100, 1);

display_network(out_real_patches(:, rn), 12);