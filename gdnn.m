function [bestW, hyperParam] = gdnn(deviceID, ptParamName, hyperParam, dataName, modelName)
if(isempty(deviceID))
    deviceID = 1;
end
dev = gpuDevice(deviceID);
% INPUT:
% ptParamName: name of pretrained hyperparameters files
% hyperParam: hyperParameters
% dataName: name of training, heldout and testing data
% modelName: prefix of names of models to save

%% step 1. set up the hyperparameter
[hyperParam, lrate, lrScale, scaleFreq, momentum, l2, nEpoch, mbSize, valFreq, earlyStop] = hyperSetting(hyperParam);

%% step 2. load training data, heldout data and optionally test data
load(dataName.nameTr);
load(dataName.nameVal);

if(min(labelTr)==0)
    labelTr = labelTr-min(labelTr)+1;
    labelVal = labelVal-min(labelVal)+1;
end

if(max(max(dataTr))>200)
    dataTr = dataTr/256;
    dataVal = dataVal/256;
end
nSampleTr = length(labelTr);
nSampleDev = length(labelVal);

boolTe = 0;
if(isfield(dataName, 'nameTe'))
    if(~isempty(dataName.nameTe))
        boolTe = 1;
        load(dataName.nameTe);
        if(max(max(dataTe))>200)
            dataTe = dataTe/256;
        end
        labelTe = labelTe-min(labelTe)+1;
        nSampleTe = length(labelTe);
    end
end

nClass = max(labelTr);
Y = sparse(double(labelTr), 1:double(nSampleTr), ones(nSampleTr, 1), double(nClass), nSampleTr); 

%% step 3. load the initial parameters
[W1, W2, W3, W4, W5, dW1, dW2, dW3, dW4, dW5, dim0, dim1, dim2, dim3, dim4] = ptParamSetting(ptParamName, nClass);


%% step 4. setup variables for loop-learning

% Xi: input to the i-th convolution
X1          = gpuArray.ones(dim0+1, mbSize, 'double');
X2          = gpuArray.zeros(dim1+1, mbSize, 'double');
X3          = gpuArray.zeros(dim2+1, mbSize, 'double');
X4          = gpuArray.zeros(dim3+1, mbSize, 'double');
X5          = gpuArray.zeros(dim4+1, mbSize, 'double');

% Zi:  output from the i-th convolution
Z1          = gpuArray.zeros(dim1+1, mbSize, 'double');
Z2          = gpuArray.zeros(dim2+1, mbSize, 'double');
Z3          = gpuArray.zeros(dim3+1, mbSize, 'double');
Z4          = gpuArray.zeros(dim4+1, mbSize, 'double');
Z5          = gpuArray.zeros(nClass, mbSize, 'double');
P           = gpuArray.zeros(nClass, mbSize, 'double');
batchY      = gpuArray.zeros(nClass, mbSize, 'double');

% used for backprop
delta6      = gpuArray.zeros(nClass, mbSize, 'double');
delta5      = gpuArray.zeros(dim4+1, mbSize, 'double');
delta4      = gpuArray.zeros(dim3+1, mbSize, 'double');
delta3      = gpuArray.zeros(dim2+1, mbSize, 'double');
delta2      = gpuArray.zeros(dim1+1, mbSize, 'double');

% used for backprop
H2          = gpuArray.zeros(dim1+1, mbSize, 'double');
H3          = gpuArray.zeros(dim2+1, mbSize, 'double');
H4          = gpuArray.zeros(dim3+1, mbSize, 'double');
H5          = gpuArray.zeros(dim4+1, mbSize, 'double');

bestW{1} = W1;
bestW{2} = W2;
bestW{3} = W3;
bestW{4} = W4;
bestW{5} = W5;

gW1 = gpuArray(single(W1));
gdW1 = gpuArray(single(dW1));
gW2 = gpuArray(single(W2));
gdW2 = gpuArray(single(dW2));
gW3 = gpuArray(single(W3));
gdW3 = gpuArray(single(dW3));
gW4 = gpuArray(single(W4));
gdW4 = gpuArray(single(dW4));
gW5 = gpuArray(single(W5));
gdW5 = gpuArray(single(dW5));



best_dev = 0;
best_te = 0;
best_batch = 0; % indicator if best heldout acc is updated
iter_count = 0;
ES_count = 0;
bool_break = 0;

%% step 5: loop learning
for epoch = 1:nEpoch
    %index = randperm(nSampleTr);
    index = 1:nSampleTr;
    % create a mini-batch of training data set, 50 samples
    for firstIdx = 1:mbSize:nSampleTr
        
        lastIdx = min(firstIdx+mbSize-1, nSampleTr);
        batchSize = lastIdx - firstIdx + 1;
        
        if(batchSize==mbSize)
        iter_count = iter_count+1;
        if(rem(iter_count, scaleFreq)==0)
            lrate = lrate*lrScale;
        end
        

        X1(2:end,1:batchSize) = dataTr(:,index(firstIdx:lastIdx));
        X1(1,1:batchSize) = X1(1,1:batchSize)*0+1; % for safety
        batchY(:,1:batchSize) = single(full(Y(:,index(firstIdx:lastIdx))));
        
        % forward propagation 
        % feed-forward-1
        Z1 = gW1'*X1;
        X2 = 1./(1+exp(-Z1));       % n_hu1 x mbSize
        
        % feed-forward-2
        Z2 = gW2'*X2;                % n_hu2 x mbSize
        X3 = 1./(1+exp(-Z2));

        % feed-forward-2
        Z3 = gW3'*X3;                % n_hu3 x mbSize
        X4 = 1./(1+exp(-Z3));

        % feed-forward-2
        Z4 = gW4'*X4;                % n_hu4 x mbSize
        X5 = 1./(1+exp(-Z4));

        % feed-forward-3
        Z5 = gW5'*X5;                % nClass x mbSize matrix

        P = bsxfun(@minus, Z5, max(Z5));
        P = exp(P);
        P = bsxfun(@rdivide, P, sum(P));

        % backward-5
        delta6 = P-batchY;              % 1000 x mbSize

        % backward-4
        H5 = X5.*(1-X5);            % 501 x mbSize
        delta5 = H5.*(gW5*delta6);   % 501 x mbSize

        % backward-3
        H4 = X4.*(1-X4);            % 501 x mbSize
        delta4 = H4.*(gW4*delta5);   % 501 x mbSize

        % backward-2
        H3 = X3.*(1-X3);            % 501 x mbSize
        delta3 = H3.*(gW3*delta4);   % 501 x mbSize

        % backward-1
        H2 = X2.*(1-X2);            % 501 x mbSize
        delta2 = H2.*(gW2*delta3);   % 501 x mbSize

        % final expression of delta-W
        gdW5 = lrate*X5(:,1:batchSize)*transpose(delta6(:,1:batchSize))/batchSize + momentum*gdW5 + l2*gW5; % 501x1000
        gdW4 = lrate*X4(:,1:batchSize)*transpose(delta5(:,1:batchSize))/batchSize + momentum*gdW4 + l2*gW4; % 501x1000
        gdW3 = lrate*X3(:,1:batchSize)*transpose(delta4(:,1:batchSize))/batchSize + momentum*gdW3 + l2*gW3; % 501x1000
        gdW2 = lrate*X2(:,1:batchSize)*transpose(delta3(:,1:batchSize))/batchSize + momentum*gdW2 + l2*gW2; % 501x501
        gdW1 = lrate*X1(:,1:batchSize)*transpose(delta2(:,1:batchSize))/batchSize + momentum*gdW1 + l2*gW1; % 361x501

        %keyboard
        % question: should lrate multiple the weight decay term?
        % obj is to minimize the neg-log-likelihood
        gW5 = gW5 - gdW5;
        gW4 = gW4 - gdW4;
        gW3 = gW3 - gdW3;
        gW2 = gW2 - gdW2;
        gW1 = gW1 - gdW1;

        % important: reset the extra dimension of W back to be 0
        gW1(:,1) = gW1(:,1)*0;
        gW2(:,1) = gW2(:,1)*0;
        gW3(:,1) = gW3(:,1)*0;
        gW4(:,1) = gW4(:,1)*0;
        

        if(rem(iter_count,valFreq)==0)
            acc_dev = 0;
            % evaluation on the heldout data
            for startIdx = 1:mbSize:nSampleDev
                endIdx = min(startIdx+mbSize-1, nSampleDev);
                batchSize = endIdx - startIdx + 1;

                X1(2:end,1:batchSize) = dataVal(:,startIdx:endIdx);
                X1(1,1:batchSize) = X1(1,1:batchSize)*0+1; % for safety
                y1 = labelVal(startIdx:endIdx);
        
                % feed-forward-1
                Z1 = gW1'*X1;
                X2 = 1./(1+exp(-Z1));       % 501xmbSize
            
                % feed-forward-2
                Z2 = gW2'*X2;                % 501xmbSize
                X3 = 1./(1+exp(-Z2));

                % feed-forward-2
                Z3 = gW3'*X3;                % 501xmbSize
                X4 = 1./(1+exp(-Z3));

                % feed-forward-2
                Z4 = gW4'*X4;                % 501xmbSize
                X5 = 1./(1+exp(-Z4));

                % feed-forward-3
                Z5 = gW5'*X5;                % 1000xmbSize matrix
                
                [~, pred] = max(Z5(:,1:batchSize));
                acc_dev = acc_dev+sum(pred(:)==y1(:));
            end
            acc_dev = gather(acc_dev);
            fprintf('epoch %d, overall batch %d, heldout accuracy %f, best heldout accuracy %f', epoch, iter_count, acc_dev/nSampleDev, best_dev);
            
            if(acc_dev/nSampleDev>=best_dev)
                best_dev = acc_dev/nSampleDev;
                best_batch = 1;
                bestW{1} = gather(gW1);
                bestW{2} = gather(gW2);
                bestW{3} = gather(gW3);
                bestW{4} = gather(gW4);
                bestW{5} = gather(gW5);
                ES_count = 0;
            else
                best_batch = 0;
                ES_count = ES_count+1;
                if(ES_count>earlyStop)
                    bool_break = 1;
                    break;
                end
            end
            
            acc_te = 0;
            if(boolTe==1)
                % evaluation on the heldout data
                for startIdx = 1:mbSize:nSampleTe
                    endIdx = min(startIdx+mbSize-1, nSampleTe);
                    batchSize = endIdx - startIdx + 1;

                    X1(2:end,1:batchSize) = dataTe(:,startIdx:endIdx);
                    y1 = labelTe(startIdx:endIdx);

                    % feed-forward-1
                    Z1 = gW1'*X1;
                    X2 = 1./(1+exp(-Z1));   

                    % feed-forward-2
                    Z2 = gW2'*X2;             
                    X3 = 1./(1+exp(-Z2));

                    % feed-forward-2
                    Z3 = gW3'*X3;              
                    X4 = 1./(1+exp(-Z3));

                    % feed-forward-2
                    Z4 = gW4'*X4;               
                    X5 = 1./(1+exp(-Z4));

                    % feed-forward-3
                    Z5 = gW5'*X5;                

                    [~, pred] = max(Z5(:,1:batchSize));
                    acc_te = acc_te+sum(pred(:)==y1(:));
                end
                acc_te = gather(acc_te);
                fprintf(', testing accuracy %f\n', acc_te/nSampleTe);
                if(best_batch==1)
                    best_te = acc_te/nSampleTe;
                end
            else
                fprintf('\n');
            end
        end
    end
    end
    if(bool_break==1)
        break;
    end
    epochName = [modelName '_epoch' num2str(epoch) '.mat'];
    save(epochName, 'bestW', 'epoch', 'hyperParam','best_dev','best_te','ptParamName');
end
modelName = [modelName '.mat'];
save(modelName, 'bestW', 'hyperParam','best_dev','best_te','ptParamName');
end

function [hyperParam, lrate, lrScale, scaleFreq, momentum, l2, nEpoch, mbSize, valFreq, earlyStop] = hyperSetting(hyperParam)
% set up the hyperparameter

% case 1. 
if(isempty(hyperParam))
    % if hyperparameters are not manually specified, use default values

    lrate = 0.08;
    lrScale = 0.9;
    scaleFreq = 1000; % lrate = lrate*lrScale every scaleFreq mini-batch
    
    momentum = 0.9;
    l2 = 0.00002; 

    nEpoch = 5;
    mbSize = 50;
    valFreq = 1000; % every valFreq minibatch, calculate the validation&testing accuracy
    earlyStop = 5; % if earlyStop*valFreq minibatch does not improve validation acc, stop
else

    if(isfield(hyperParam, 'lrate'))
        lrate = hyperParam.lrate;
    else
        lrate = 0.08;
    end

    % lrate = lrate*lrScale every scaleFreq mini-batch
    if(isfield(hyperParam, 'lrScale'))
        lrScale = hyperParam.lrScale;
    else
        lrScale = 0.9;
    end

    % lrate = lrate*lrScale every scaleFreq mini-batch
    if(isfield(hyperParam, 'scaleFreq'))
        scaleFreq = hyperParam.scaleFreq;
    else
        scaleFreq = 1000;
    end

    if(isfield(hyperParam, 'momentum'))
        momentum = hyperParam.momentum;
    else
        momentum = 0.9;
    end

    if(isfield(hyperParam, 'l2'))
        l2 = hyperParam.l2;
    else
        l2 = 0.00002;
    end

    if(isfield(hyperParam, 'nEpoch'))
        nEpoch = hyperParam.nEpoch;
    else
        nEpoch = 15;
    end

    if(isfield(hyperParam, 'mbSize'))
        mbSize = hyperParam.mbSize;
    else
        mbSize = 50;
    end

    % every valFreq more minibatchs, calculate the validation&testing accuracy
    if(isfield(hyperParam, 'valFreq'))
        valFreq = hyperParam.valFreq;
    else
        valFreq = 1000;
    end

    % if earlyStop*valFreq minibatch does not improve validation acc, stop
    if(isfield(hyperParam, 'earlyStop'))
        earlyStop = hyperParam.earlyStop;
    else
        earlyStop = 5;
    end

    hyperParam.lrate = lrate;
    hyperParam.momentum = momentum;
    hyperParam.lrScale = lrScale;
    hyperParam.scaleFreq = scaleFreq;
    hyperParam.valFreq = valFreq;
    hyperParam.l2 = l2;
    hyperParam.nEpoch = nEpoch;
    hyperParam.earlyStop = earlyStop;
    hyperParam.mbSize = mbSize;
end
end

function [W1, W2, W3, W4, W5, dW1, dW2, dW3, dW4, dW5, dim0, dim1, dim2, dim3, dim4] = ptParamSetting(ptParamName, nClass)

    % by default, the pretrained parameters are of the following dimensionality:
    %   W1: dim0 x dim1, hidBias1: dim1 x 1
    %   W2: dim1 x dim2, hidBias2: dim2 x 1
    % ...
    %   W5: dim4 x nClass, hidBias5: nClass x 1
    % where network size is dim0 --> dim1 --> dim2 --> dim3 --> dim4 --> nClass
    
    ptLayer1 = ptParamName{1}; % e.g. 'model1_ptL1NN.mat';
    ptLayer2 = ptParamName{2}; % e.g. 'model1_ptL2NN.mat';
    ptLayer3 = ptParamName{3}; % e.g. 'model1_ptL3NN.mat';
    ptLayer4 = ptParamName{4}; % e.g. 'model1_ptL4NN.mat';

    load(ptLayer1); % --> W1 and hidBias1
    load(ptLayer2); % --> W2 and hidBias2
    load(ptLayer3); % --> W3 and hidBias3
    load(ptLayer4); % --> W4 and hidBias4

    % W5, randomly assigned, is preferably provided, 
    % so that the comparison between different hyperparameters makes more sense
    W5 = [];
    hidBias5 = [];
    if(length(ptParamName)>4)
        ptLayer5 = ptParamName{5}; % e.g. 'model1_ptL5NN.mat';
        load(ptLayer5);
    end
    
    % preprocessing the pretrained data
    % 1. combined W and hidBias into one matrix
    % 2. add one more output (upper layer) dimension (all 0s)

    if(size(W1,1)~=length(hidBias1))
        W1 = W1';
        W2 = W2';
        W3 = W3';
        W4 = W4';
        W5 = W5';
    end

    if(size(hidBias1,1)==1)
        hidBias1 = hidBias1';
        hidBias2 = hidBias2';
        hidBias3 = hidBias3';
        hidBias4 = hidBias4';
        hidBias5 = hidBias5';
    end

    dim0 = size(W1,2);
    W1 = [hidBias1 W1];   % 1. concatenate bias and W
    W1 = [W1(1,:)*0; W1]; % 2. add one constant unit toward output layer
    W1 = W1';
    dW1 = W1*0;

    dim1 = size(W2,2);
    W2 = [2*hidBias2 W2]; 
    W2 = [W2(1,:)*0; W2]; 
    W2 = W2';
    dW2 = W2*0;

    dim2 = size(W3,2);
    W3 = [2*hidBias3 W3]; 
    W3 = [W3(1,:)*0; W3]; 
    W3 = W3';
    dW3 = W3*0;

    dim3 = size(W4,2);
    dim4 = size(W4,1);
    W4 = [2*hidBias4 W4]; 
    W4 = [W4(1,:)*0; W4]; 
    W4 = W4';
    dW4 = W4*0;

    if(isempty(W5))
        W5 = (rand(nClass, dim4+1) - 0.5) * 2 * 4 * sqrt(6 / (10 + 2001));
        %W5 = (rand(nClass, dim4+1,'double') - 0.5) * 2 * 4 * sqrt(6 / (10 + 2001));
    else
        W5 = [2*hidBias5 W5]; 
    end
    W5 = W5';
    dW5 = W5*0;

    % note-->
    % in this function, we did multiple transpose of W matrix
    % to be clear, the output W is of such format (take W1 and MNIST for examle):
    %   W1: 1001x361
    %       first column are all 0-elements, corresponding to a constant node in upper layer
    %       first row, from column2 to column_end are hidBias parameters
    % <--note
end
