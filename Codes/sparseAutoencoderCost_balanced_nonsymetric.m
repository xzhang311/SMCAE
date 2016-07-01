function [cost,grad] = sparseAutoencoderCost_balanced_nonsymetric(theta, visibleSize, hiddenSize, outputSize, ...
                                             lambda, sparsityParam, balanceParam, beta, ldata_in, rdata_in, ldata_out, rdata_out)

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

% wLen = hiddenSize*visibleSize;
% bLen1 =  hiddenSize;
% bLen2 =  visibleSize;
% 
% W1 = reshape(theta(1 : wLen), hiddenSize, visibleSize);
% lW2 = reshape(theta(wLen+1 : 2*wLen), visibleSize, hiddenSize);
% rW2 = reshape(theta(2*wLen+1 : 3*wLen), visibleSize, hiddenSize);
% b1 = theta(3*wLen+1 : 3*wLen+bLen1);
% lb2 = theta(3*wLen+bLen1+1 : 3*wLen+bLen1+bLen2);
% rb2 = theta(3*wLen+bLen1+bLen2+1 : end);

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
lz3 = lW2 * la2 + repmat(lb2, 1, lm);
la3 = sigmoid(lz3);

% forward pass for right branch
rz2 = W1 * rdata_in + repmat(b1, 1, rm);
ra2 = sigmoid(rz2);
rz3 = rW2 * ra2 + repmat(rb2, 1, rm);
ra3 = sigmoid(rz3);

rho = (sum(la2, 2)+sum(ra2, 2)) / (lm+rm);    

% left cost
ldiff = la3 - ldata_out;
sparse_penalty = kl(sparsityParam, rho);
lJ_simple = sum(sum(ldiff.^2)) / (2*lm);
reg = sum(W1(:).^2) + sum(lW2(:).^2);
lcost = lJ_simple + beta * sparse_penalty + lambda * reg / 2;

% right cost
rdiff = ra3 - rdata_out;
sparse_penalty = kl(sparsityParam, rho);
rJ_simple = sum(sum(rdiff.^2)) / (2*rm);
reg = sum(W1(:).^2) + sum(rW2(:).^2);
rcost = rJ_simple + beta * sparse_penalty + lambda * reg / 2;

% compute propagate error of left branch
ldelta3 = ldiff .* (la3 .* (1 - la3));
ld2_simple = lW2' * ldelta3;
ld2_pen = kl_delta(sparsityParam, rho);
ldelta2 = (ld2_simple + beta * repmat(ld2_pen, 1, lm)) .* la2 .* (1-la2);

% compute propagate error of right branch
rdelta3 = rdiff .* (ra3 .* (1 - ra3));
rd2_simple = rW2' * rdelta3;
rd2_pen = kl_delta(sparsityParam, rho);
rdelta2 = (rd2_simple + beta * repmat(rd2_pen, 1, rm)) .* ra2 .* (1-ra2);

% gradient of parameters in second layer
lW2grad_simple = ldelta3 * la2'/lm + lambda * lW2;
lb2grad_simple = sum(ldelta3 , 2)/lm;
rW2grad_simple = rdelta3 * ra2'/lm + lambda * rW2;
rb2grad_simple = sum(rdelta3 , 2)/rm;

lW2grad = lW2grad_simple + balanceParam * (lcost - rcost) * lW2grad_simple;
lb2grad = lb2grad_simple + balanceParam * (lcost - rcost) * lb2grad_simple;
rW2grad = rW2grad_simple + balanceParam * (lcost - rcost) * -1 * rW2grad_simple;
rb2grad = rb2grad_simple + balanceParam * (lcost - rcost) * -1 * rb2grad_simple;

% gradient of parameters in first layer
lW1grad = ldelta2 * ldata_in' / lm +lambda * W1;
lb1grad = sum(ldelta2, 2) / lm;
rW1grad = rdelta2 * rdata_in'/rm + lambda * W1;
rb1grad = sum(rdelta2, 2)/rm;

W1grad = lW1grad + rW1grad + balanceParam * (lcost - rcost) * (lW1grad - rW1grad);
b1grad = lb1grad + rb1grad + balanceParam * (lcost - rcost) * (lb1grad - rb1grad);

% final cost
cost = lcost + rcost + balanceParam * 0.5*(lcost - rcost)^2;

% form gradient vector for fminfunc
grad = [W1grad(:) ; lW2grad(:) ; rW2grad(:) ; b1grad(:) ; lb2grad(:) ; rb2grad(:)];

% print information
lcost_str=num2str(lcost);
rcost_str=num2str(rcost);
first_term_str=num2str(lcost+rcost);
second_term_str=num2str(balanceParam*0.5*(lcost-rcost)^2);

mystr=['left: ', lcost_str, ' right: ', rcost_str, ' 1st term cost: ', first_term_str, ' 2nd term cost: ', second_term_str];
disp(mystr);
end

function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

function ans = kl(r, rh)
    ans = sum(r .* log(r ./ rh) + (1-r) .* log( (1-r) ./ (1-rh)));
end

function ans = kl_delta(r, rh)
    ans = -(r./rh) + (1-r) ./ (1-rh);
end

