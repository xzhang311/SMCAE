function theta = initializeParametersNonsymetric(hiddenSize, visibleSize, outputSize)

%% Initialize parameters randomly based on layer sizes.
r  = sqrt(6) / sqrt(hiddenSize+visibleSize+1);   % we'll choose weights uniformly from the interval [-r, r]
W1 = rand(hiddenSize, visibleSize) * 2 * r - r;
lW2 = rand(outputSize, hiddenSize) * 2 * r - r;
rW2 = rand(outputSize, hiddenSize) * 2 * r -r;

b1 = zeros(hiddenSize, 1);
lb2 = zeros(outputSize, 1);
rb2 = zeros(outputSize, 1);

% Convert weights and bias gradients to the vector form.
% This step will "unroll" (flatten and concatenate together) all 
% your parameters into a vector, which can then be used with minFunc. 
theta = [W1(:) ; lW2(:) ; rW2(:) ; b1(:) ; lb2(:) ; rb2(:)];

end

