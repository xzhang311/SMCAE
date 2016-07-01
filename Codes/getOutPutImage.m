function [ outimg, features ] = getOutPutImage( img, theta, visibleSize, hiddenSize, imageSize )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

[ndims, m] = size(img);

z2 = zeros(hiddenSize, m);
z3 = zeros(visibleSize, m);
a1 = zeros(ndims, m);
a2 = zeros(size(z2));
a3 = zeros(size(z3));

%forward pass 
for i = 1 : m
    z2(:,i) = W1 * img(:,i) + b1;
    a2(:,i) = sigmoid(z2(:,i));
    z3(:,i) = W2 * a2(:,i) + b2;
    a3(:,i) = sigmoid(z3(:,i));
end

outimg=reshape(a3, imageSize, imageSize, m);
features=a2;

end

