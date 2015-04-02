function [model] = train_net_gpu(x,y,weight,hypers)

% train a one hidden layer neural nets 
% on \{(x_i,y_i)\}_n ~ weight

[n,d,K,H,n_epoch,batch_size,l_rate,device] = getHypers(hypers);

unif_value = sqrt(6/(d+K));
V = unifrnd(-unif_value,unif_value,H,d);  % e.g. d = 784+1 = 785

W = zeros(K,H+1);
% Z1 = ones(H,batch_size);
% X2 = ones(H+1,batch_size);

n_batch = ceil(n / batch_size);
% x_batch = zeros(d,batch_size);
% y_batch = zeros(K,batch_size);

Y = sparse(double(y), 1:double(n), ones(n, 1), double(K), n); 
prediction = zeros(1,n);

batch_size_eval = 5000;
n_batch_eval = ceil(n/batch_size_eval);

% x_batch_eval = zeros(d,batch_size_eval);
% Z1_eval = ones(H,batch_size_eval);
% X2_eval = ones(H+1,batch_size_eval);

dev = gpuDevice(device);
hyperindex = 1;
parallel.gpu.rng(hyperindex);
fprintf('\nGPU Detected (%s, %d multiprocessors, Compute Capability %s, Mem %d) \n',...
    dev.Name, dev.MultiprocessorCount, dev.ComputeCapability, dev.FreeMemory);

gV = gpuArray(single(V));
gW = gpuArray(single(W));

gX_batch = gpuArray.zeros(d,batch_size);
gY_batch = gpuArray.zeros(K,batch_size);
gZ1 = gpuArray.ones(H,batch_size);
gX2 = gpuArray.ones(H+1,batch_size);
% P = gpuArray.zeros(K, batch_size);
% delta3 = gpuArray.zeros(K,batch_size);
% delta2 = gpuArray.zeros(H+1,batch_size);
% gH2 = gpuArray.zeros(H+1,batch_size);

gZ1_eval = gpuArray.ones(H,batch_size_eval);
gX2_eval = gpuArray.ones(H+1,batch_size_eval);
gX_batch_eval = gpuArray.zeros(d,batch_size_eval);
% gY_batch_eval = gpuArray.zeros(K,batch_size_eval);

err_tr = 0;
% acc_train = 0;
% i = 0;
% while (i <= n_epoch) ||  (acc_train < 0.7) 
for i = 1:n_epoch
%     i = i+1;
%     if i > 50
%         break
%     end
    R = mnrnd(n,weight,1);
    index = sample2ind(R,n);
    index = index(randperm(n));
%     index = randperm(n);

    for j = 1:n_batch
        firstInd = 1+(j-1) * batch_size;
        if j ~= n_batch
            lastInd = j * batch_size;
        else
            lastInd = n;
        end
        mb_size = lastInd - firstInd + 1;
        
%         x_batch(:,1:mb_size) = x(:,index(firstInd:lastInd)); % d by n
        gX_batch(:,1:mb_size) = x(:,index(firstInd:lastInd));
        gY_batch(:,1:mb_size) = full(Y(:,index(firstInd:lastInd)));

        % feedforward
%         Z1(:,1:mb_size) = V * x_batch(:,1:mb_size);  % Z1: (1+H) by n
        gZ1(:,1:mb_size) = gV * gX_batch(:,1:mb_size);
%         X2(2:end,1:mb_size) = 1./(1+exp(-Z1(:,1:mb_size)));   % X2 same as Z1)
        gX2(2:end,1:mb_size) = 1./(1+exp(-gZ1(:,1:mb_size)));
        
%         Z2 = W * X2;
        gZ2 = gW * gX2(2:end,1:mb_size);
        
        P = bsxfun(@minus,gZ2,max(gZ2));
        P = exp(P);
        P = bsxfun(@rdivide,P,sum(P));
        
        delta3 = P - gY_batch(:,1:mb_size);  % K by n
        
        gH2 = gX2 .* (1-gX2); %H+1 by n
        delta2 = gH2.*(gW' * delta3);  % (H+1) by n
        
        w_batch = weight(index(firstInd:lastInd));
        w_batch = w_batch / sum(w_batch);
        
        delta3 = delta3 .* repmat(w_batch,K,1);
        delta2 = delta2 .* repmat(w_batch,H+1,1);
        
        grad_W = delta3(:,1:mb_size) * transpose(gX2(:,1:mb_size))  ;
        grad_V = delta2(2:end,1:mb_size) * transpose(gX_batch(:,1:mb_size)) ;

        gW = gW - grad_W * l_rate;
        gV = gV - grad_V * l_rate;
%        fprintf('norm of W,gradW,V,gradV: %f,%f; %f,%f\n',norm(W),norm(grad_W),norm(V), norm(grad_V));
    end
    
    % evaluation
    acc_train = 0;
    for j = 1:n_batch_eval
        firstInd = 1+(j-1) * batch_size_eval;
        if j ~= n_batch
            lastInd = j * batch_size_eval;
        else
            lastInd = n;
        end
        mb_size = lastInd - firstInd + 1;
        
        gX_batch_eval(:,1:mb_size) = x(:,firstInd:lastInd); % d by n
        yb = y(firstInd:lastInd);
        
        % feedforward
        gZ1_eval(:,1:mb_size) = V * gX_batch_eval(:,1:mb_size);  % Z1: (1+H) by n
        gX2_eval(2:end,1:mb_size) = 1./(1+exp(-gZ1_eval(:,1:mb_size)));   % X2 same as Z1)
        gZ2_eval = gW * gX2_eval;

        [~,pred] = max(gZ2_eval(:,1:mb_size));
        w_batch = weight(firstInd:lastInd);
        if i == n_epoch
            prediction(firstInd:lastInd) = pred;
        end
%         acc_train = acc_train + sum(pred(:)==yb(:));
        acc_train = acc_train + sum((pred(:)==yb(:)).*w_batch');
    end
%     if i == n_epoch
%         fprintf('epoch %d, training acc %f\n',i,acc_train);
%     end
    err_tr = 1 - acc_train;
end

model.pred = gather(prediction);
model.err_tr = gather(err_tr);
model.alpha = gather(log(1/err_tr - 1) + log(K-1));
model.V = gather(gV);
model.W = gather(gW);

fprintf('Training err: %f; alpha: %f\n',model.err_tr,model.alpha)



function [n,d,K,H,n_epoch,batch_size,l_rate,device] = getHypers(hypers)
device = hypers.device;
n = hypers.n_train;
d = hypers.dim;
K = hypers.no_classes;
H = hypers.n_hid;
n_epoch = hypers.nIters;
batch_size = hypers.batch_size;
l_rate = hypers.learning_rate;