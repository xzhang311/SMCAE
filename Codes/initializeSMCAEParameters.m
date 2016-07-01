function theta = initializeSMCAEParameters(SMCAE)
% Form a long array for parameters Ws and bs of entire SMCAE
% Start from input layer to output layer.
% For all layers except output layer, only store one W and one b.
% For output layer, store Ws and bs for left and right branches separately.
    theta=[];
    for i=1:length(SMCAE)
        theta = [theta ; SMCAE(i).W1(:) ; SMCAE(i).b1(:)];
        
        if i==length(SMCAE)
            theta = [theta ; SMCAE(i).lW2(:) ; SMCAE(i).rW2(:) ; SMCAE(i).lb2(:) ; SMCAE(i).rb2(:)];
        end
    end
end