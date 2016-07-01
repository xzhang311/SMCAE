real_path = '/home/xi/Project/LearningSyntheticRoof/data/Mine/Training/Flipped_Noised/Real/pyramid/real.mat';
tmplt_path = '/home/xi/Project/LearningSyntheticRoof/data/Mine/Training/Flipped_Noised/Template/pyramid/tmplt.mat';
out_real_path = '/home/xi/Project/LearningSyntheticRoof/data/Mine/Training/Flipped_Noised/Real/pyramid/real_100.mat';
out_tmplt_path = '/home/xi/Project/LearningSyntheticRoof/data/Mine/Training/Flipped_Noised/Template/pyramid/tmplt_100.mat';

real_patches=load(real_path);
tmplt_patches=load(tmplt_path);

real_100=zeros(100, 128, 256);
tmplt_100=zeros(100, 128, 256);

for i=1:100
    id=randi([1, size(real_patches.allImages, 1)], 1,1 );
    real_img=reshape(real_patches.allImages(id, :, :), 128, 256);
    tmplt_img=reshape(tmplt_patches.allTmplts(id, :, :), 128, 256);
    real_100(i, :, :)=real_img;
    tmplt_100(i, :, :)=tmplt_img;
end

save(out_real_path, 'real_100');
save(out_tmplt_path, 'tmplt_100');