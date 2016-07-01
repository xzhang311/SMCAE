function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, ldata_in, rdata_in, ldata_out, rdata_out)

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

wLen = hiddenSize*visibleSize;
bLen1 =  hiddenSize;
bLen2 =  visibleSize;

W1 = reshape(theta(1 : wLen), hiddenSize, visibleSize);
lW2 = reshape(theta(wLen+1 : 2*wLen), visibleSize, hiddenSize);
rW2 = reshape(theta(2*wLen+1 : 3*wLen), visibleSize, hiddenSize);
b1 = theta(3*wLen+1 : 3*wLen+bLen1);
lb2 = theta(3*wLen+bLen1+1 : 3*wLen+bLen1+bLen2);
rb2 = theta(3*wLen+bLen1+bLen2+1 : end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(lW2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(lb2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 
[ndims, m] = size(ldata_in);
[ndims, rm] = size(rdata_in);

lz2 = zeros(hiddenSize, m); rz2 = zeros(hiddenSize, rm);
lz3 = zeros(visibleSize, m); rz3 = zeros(visibleSize, rm);
la1 = zeros(ndims, m); ra1 = zeros(ndims, rm);
la2 = zeros(size(lz2)); ra2 = zeros(size(rz2));
la3 = zeros(size(lz3)); ra3 = zeros(size(rz3));

%autoencode use inputs as target values
ly = zeros(ndims, m); ry = zeros(ndims, rm);

la1 = ldata_in; ra1 = rdata_in;
ly = ldata_out; ry = rdata_out;

%%%% We need to change here to get our method run
deltaW1 = zeros(size(W1));
deltab1 = zeros(size(b1));
JW1grad = zeros(size(W1));
Jb1grad = zeros(size(b1));

ldeltaW2 = zeros(size(lW2)); rdeltaW2 = zeros(size(lW2));
ldeltab2 = zeros(size(lb2)); rdeltab2 = zeros(size(lb2));
lJW2grad = zeros(size(lW2)); rJW2grad = zeros(size(lW2));
lJb2grad = zeros(size(lb2)); rJb2grad = zeros(size(lb2));

%forward pass 
for i = 1 : m
    lz2(:,i) = W1 * ldata_in(:,i) + b1; rz2(:,i) = W1 * rdata_in(:,i) + b1;
    la2(:,i) = sigmoid(lz2(:,i)); ra2(:,i) = sigmoid(rz2(:,i));
    lz3(:,i) = lW2 * la2(:,i) + lb2; rz3(:,i) = rW2 * ra2(:,i) + rb2;
    la3(:,i) = sigmoid(lz3(:,i)); ra3(:,i) = sigmoid(rz3(:,i));
end

rho = zeros(hiddenSize, 1);
rho = (sum(la2, 2)+sum(ra2, 2)) / (m+rm);    
sp = sparsityParam;

for i = 1 : m
    ldelta3 = -(ly(:,i) - la3(:,i)) .* sigmoidGrad(lz3(:,i)); 
    rdelta3 = -(ry(:,i) - ra3(:,i)) .* sigmoidGrad(rz3(:,i));
    
    ldelta2 = ( lW2' * ldelta3 + beta * (-sp ./ rho + (1-sparsityParam) ./ (1-rho) ) ) ...
        .* sigmoidGrad(lz2(:,i));
    rdelta2 = ( rW2' * rdelta3 + beta * (-sp ./ rho + (1-sparsityParam) ./ (1-rho) ) ) ...
        .* sigmoidGrad(rz2(:,i));
    
    lJW2grad = ldelta3 * la2(:,i)'; rJW2grad = rdelta3 * ra2(:,i)';
    lJb2grad = ldelta3; rJb2grad = rdelta3;
    
    JW1grad = ldelta2 * la1(:,i)' + rdelta2 * ra1(:, i)';  %%%%%% Need double check
    Jb1grad = ldelta2 + rdelta2;
    
    ldeltaW2 = ldeltaW2 + lJW2grad; rdeltaW2 = rdeltaW2 + rJW2grad;
    ldeltab2 = ldeltab2 + lJb2grad; rdeltab2 = rdeltab2 + rJb2grad;
    deltaW1 = deltaW1 + JW1grad;
    deltab1 = deltab1 + Jb1grad;
end


W1grad = (1. / m) * deltaW1 + lambda * W1;
b1grad = (1. / m) * deltab1;
lW2grad = (1. / m) * ldeltaW2 + lambda * lW2; rW2grad = (1. / m) * rdeltaW2 + lambda * rW2;
lb2grad = (1. / m) * ldeltab2; rb2grad = (1. / m) * rdeltab2;

lcost = (1. / m) * sum((1. / 2) * sum((la3 - ly).^2)) + ...
    (lambda / 2.) * (sum(sum(W1.^2)) + sum(sum(lW2.^2))) + ...
    beta * sum( sp*log(sp./rho) + (1-sp)*log((1-sp)./(1-rho)) );
rcost = (1. / m) * sum((1. / 2) * sum((ra3 - ry).^2)) + ...
    (lambda / 2.) * (sum(sum(W1.^2)) + sum(sum(rW2.^2))) + ...
    beta * sum( sp*log(sp./rho) + (1-sp)*log((1-sp)./(1-rho)) );

cost=lcost+rcost;

function grad = sigmoidGrad(x)
    e_x = exp(-x);
    grad = e_x ./ ((1 + e_x).^2); 
end


%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; lW2grad(:) ; rW2grad(:) ; b1grad(:) ; lb2grad(:) ; rb2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

