function [cost, grad] = SMCAE_Cost(theta, SMCAE)
% For fine tuning of the stacked auto-encoder, build long array of gradient
% of all variants in the SMCAE.
    nLayers = length(SMCAE);

    [ndims, lm] = size(SMCAE(1).inSynData);
    [ndims, rm] = size(SMCAE(1).inActData);

%%
% Update input and output data value in SMCAE using current theta
    thetaIdx=1;
    
    for i=1 : nLayers
        wLen = length(SMCAE(i).W1(:));
        SMCAE(i).W1 = reshape(theta(thetaIdx : thetaIdx+wLen-1), SMCAE(i).hiddenSize, SMCAE(i).visibleSize);
        thetaIdx = thetaIdx+wLen;
        bLen = length(SMCAE(i).b1(:));
        SMCAE(i).b1 = theta(thetaIdx : thetaIdx+bLen-1);
        thetaIdx = thetaIdx+bLen;
        
        [ndims, lm] = size(SMCAE(i).inSynData);
        [ndims, rm] = size(SMCAE(i).inActData);
        
        lz2 = SMCAE(i).W1 * SMCAE(i).inSynData + repmat(SMCAE(i).b1, 1, lm);
        la2 = sigmoid(lz2);
        rz2 = SMCAE(i).W1 * SMCAE(i).inActData + repmat(SMCAE(i).b1, 1, rm);
        ra2 = sigmoid(rz2);
        
        if i==nLayers
            lw2Len = length(SMCAE(i).lW2(:));
            SMCAE(i).lW2 = reshape(theta(thetaIdx : thetaIdx+lw2Len-1), SMCAE(i).outputSize, SMCAE(i).hiddenSize);
            thetaIdx = thetaIdx+lw2Len;
            rw2Len = length(SMCAE(i).rW2(:));
            SMCAE(i).rW2 = reshape(theta(thetaIdx : thetaIdx+rw2Len-1), SMCAE(i).outputSize, SMCAE(i).hiddenSize);
            thetaIdx = thetaIdx+rw2Len;
            lb2Len = length(SMCAE(i).lb2(:));
            SMCAE(i).lb2 = theta(thetaIdx : thetaIdx+lb2Len-1);
            thetaIdx = thetaIdx+lb2Len;
            rb2Len = length(SMCAE(i).rb2(:));
            SMCAE(i).rb2 = theta(thetaIdx : thetaIdx+rb2Len-1);
            
            lz3 = SMCAE(i).lW2 * la2 + repmat(SMCAE(i).lb2, 1, lm);
            la3 = sigmoid(lz3);
            rz3 = SMCAE(i).rW2 * ra2 + repmat(SMCAE(i).rb2, 1, rm);
            ra3 = sigmoid(rz3);
            SMCAE(i).predictSynData = la3;
            SMCAE(i).predictActData = ra3;
        else
            SMCAE(i+1).inSynData = la2;
            SMCAE(i+1).inActData = ra2;
        end
    end

    
%% 
% Compute cost of forward pass
% Left and right branch gets cost separately.
    lreg = 0;
    rreg = 0;
    
    for i = 1 : nLayers
        lreg = lreg + sum(SMCAE(i).W1(:).^2);
        rreg = rreg + sum(SMCAE(i).W1(:).^2);
        
        if i==nLayers
            lreg = lreg + sum(SMCAE(i).lW2(:).^2);
            rreg = rreg + sum(SMCAE(i).rW2(:).^2);
        end
    end
    
    ldiff = SMCAE(nLayers).predictSynData - SMCAE(nLayers).outSynData;
    rdiff = SMCAE(nLayers).predictActData - SMCAE(nLayers).outActData;
    lJ = sum(sum(ldiff.^2)) / (2*lm);
    rJ = sum(sum(rdiff.^2)) / (2*rm);
    lcost = lJ + SMCAE(1).lambda * lreg / 2;
    rcost = rJ + SMCAE(1).lambda * rreg / 2;
    
    upperLayerLDelta = 0;
    upperLayerRDelta = 0;
    
%%
% Back propagation of error.
% Compute gradient of all Ws and bs.
    for i = nLayers : -1 : 1
        lz2 = SMCAE(i).W1 * SMCAE(i).inSynData + repmat(SMCAE(i).b1, 1, lm);
        la2 = sigmoid(lz2);
        rz2 = SMCAE(i).W1 * SMCAE(i).inActData + repmat(SMCAE(i).b1, 1, rm);
        ra2 = sigmoid(rz2);
        
        if i == nLayers
            la3 = SMCAE(i).predictSynData;
            ra3 = SMCAE(i).predictActData;
            ldelta3 = ldiff .* (la3 .* (1 - la3));
            rdelta3 = rdiff .* (ra3 .* (1 - ra3));
            
            upperLayerLDelta = ldelta3;
            upperLayerRDelta = rdelta3;
            
            lW2grad_simple = ldelta3 * la2' / lm + SMCAE(i).lambda * SMCAE(i).lW2;
            lb2grad_simple = sum(ldelta3, 2) / lm;
            rW2grad_simple = rdelta3 * ra2' / rm + SMCAE(i).lambda * SMCAE(i).rW2;
            rb2grad_simple = sum(rdelta3, 2) / rm;
            
            SMCAE(i).lW2grad = lW2grad_simple + SMCAE(i).balanceParam * (lcost - rcost) * lW2grad_simple;
            SMCAE(i).lb2grad = lb2grad_simple + SMCAE(i).balanceParam * (lcost - rcost) * lb2grad_simple;
            SMCAE(i).rW2grad = rW2grad_simple + SMCAE(i).balanceParam * (lcost - rcost) * -1 * rW2grad_simple;
            SMCAE(i).rb2grad = rb2grad_simple + SMCAE(i).balanceParam * (lcost - rcost) * -1 * rb2grad_simple;
        end
        
        if i== nLayers
            % if last layer, use W2 from output layer.
            ld2 = SMCAE(i).lW2' * upperLayerLDelta;
            rd2 = SMCAE(i).rW2' * upperLayerRDelta;
        else
            % if not last layer, use W1 of upper layer.
            ld2 = SMCAE(i+1).W1' * upperLayerLDelta;
            rd2 = SMCAE(i+1).W1' * upperLayerRDelta;
        end
        
        ldelta2 = ld2 .* la2 .* (1-la2);
        rdelta2 = rd2 .* ra2 .* (1-ra2);
        
        upperLayerLDelta = ldelta2;
        upperLayerRDelta = rdelta2;
        
        lW1grad = ldelta2 * SMCAE(i).inSynData' / lm + SMCAE(i).lambda * SMCAE(i).W1;
        lb1grad = sum(ldelta2, 2) / lm;
        rW1grad = rdelta2 * SMCAE(i).inActData' / rm + SMCAE(i).lambda * SMCAE(i).W1;
        rb1grad = sum(rdelta2, 2) / rm; 
        
        SMCAE(i).W1grad = lW1grad + rW1grad + SMCAE(i).balanceParam * (lcost - rcost) * (lW1grad - rW1grad);
        SMCAE(i).b1grad = lb1grad + rb1grad + SMCAE(i).balanceParam * (lcost - rcost) * (lb1grad - rb1grad);
    end
    
%%
% Form gradients
    grad = [];
    for i = 1 : nLayers
        grad = [grad ; SMCAE(i).W1grad(:); SMCAE(i).b1grad(:)];
        
        if i==length(SMCAE)
            grad = [grad ; SMCAE(i).lW2grad(:); SMCAE(i).rW2grad(:); SMCAE(i).lb2grad(:); SMCAE(i).rb2grad(:)];
        end
    end
    
%%
% Compute cost
    cost = lcost + rcost + SMCAE(1).balanceParam * 0.5 * (lcost - rcost)^2;
    cost
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
