function [lout, rout] = sparseAutoencoderPredict_balanced_nonsymetric(theta, visibleSize, hiddenSize, outputSize,...
                                             lambda, sparsityParam, balanceParam, beta, ldata_in, rdata_in)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

% Structure of theta:
% _______________________________________________
% |___W1___|___lW2___|___rW2___|_b1_|_lb2_|_rb2_|

w1Len = hiddenSize * visibleSize;
w2Len = hiddenSize * outputSize;
b1Len = hiddenSize;
b2Len = outputSize;

W1 = reshape(theta(1 : w1Len), hiddenSize, visibleSize);
lW2 = reshape(theta(w1Len+1 : w1Len+w2Len), outputSize, hiddenSize);
rW2 = reshape(theta(w1Len+w2Len+1 : w1Len+2*w2Len), outputSize, hiddenSize);
b1 = theta(w1Len+2*w2Len+1 : w1Len+2*w2Len+b1Len);
lb2 = theta(w1Len+2*w2Len+b1Len+1 : w1Len+2*w2Len+b1Len+b2Len);
rb2 = theta(w1Len+2*w2Len+b1Len+b2Len+1 : w1Len+2*w2Len+b1Len+2*b2Len);


[ndims, lm] = size(ldata_in);
[ndims, rm] = size(rdata_in);

% forward pass for left branch
lz2 = W1 * ldata_in + repmat(b1, 1, lm);
la2 = sigmoid(lz2);
% lz3 = lW2 * la2 + repmat(lb2, 1, lm);
% la3 = sigmoid(lz3);
lout = la2;

% forward pass for right branch
rz2 = W1 * rdata_in + repmat(b1, 1, rm);
ra2 = sigmoid(rz2);
% rz3 = rW2 * ra2 + repmat(rb2, 1, rm);
% ra3 = sigmoid(rz3);
rout = ra2;
end

function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end


