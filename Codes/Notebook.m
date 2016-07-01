[outimg_real, features_real]=getOutPutImage(patches_real, opttheta, visibleSize, hiddenSize, image_size);
[outimg_tmplt, features_tmplt]=getOutPutImage(patches_template, opttheta, visibleSize, hiddenSize, image_size);

features=[features_real, features_tmplt];
mappedX = fast_tsne(features', 2, 30);

label_real=zeros(size(features_real, 2), 1);
label_tmplt=zeros(size(features_tmplt, 2), 1)+1;
label=[label_real;label_tmplt];

gscatter(mappedX(:,1), mappedX(:,2), label);

figure;