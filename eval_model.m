function stats = eval_model(x_tr,y_tr,x_va,y_va,N,M,stats)

V = M.V;
W = M.W;
alpha = M.alpha;

n1 = size(x_tr,2);
n2 = size(x_va,2);
batch_size_eval = 5000;
n_batch_1 = ceil(n1/batch_size_eval);
n_batch_2 = ceil(n2/batch_size_eval);

K = size(W,1);
H = size(V,1);
x_batch = zeros(size(x_tr,1),batch_size_eval);
Z1 = ones(H,batch_size_eval);
X2 = ones(H+1,batch_size_eval);

err_tr = 0;
for i = 1:n_batch_1
    firstInd = 1 + (i-1) * batch_size_eval;
    if i~=n_batch_1
        lastInd = i * batch_size_eval;
    else
        lastInd = n1;
    end
    mb_size = lastInd - firstInd+1;
    
    x_batch(:,1:mb_size) = x_tr(:,firstInd:lastInd);
    yb = y_tr(firstInd:lastInd);
    
    Z1(:,1:mb_size) = V * x_batch(:,1:mb_size);
    X2(2:end,1:mb_size) = 1./(1+exp(-Z1(:,1:mb_size)));
    Z2 = W * X2;
    
    [~,pred] = max(Z2(:,1:mb_size));
    y_batch = zeros(K,mb_size);
    indd = sub2ind([K,mb_size],pred,1:mb_size);
    y_batch(indd) = 1;
    
    
    stats.score_tr(:,firstInd:lastInd) = stats.score_tr(:,firstInd:lastInd) + y_batch * alpha;
    err_tr = err_tr +sum(yb' ~= pred);
end
err_tr = err_tr / n1;
% fprintf('%d th iter error (train): %f\n',N,err_tr);

err_va = 0;
for i = 1:n_batch_2
    firstInd = 1 + (i-1) * batch_size_eval;
    if i~=n_batch_2
        lastInd = i * batch_size_eval;
    else
        lastInd = n2;
    end
    mb_size = lastInd - firstInd + 1;
    
    x_batch(:,1:mb_size) = x_va(:,firstInd:lastInd);
    yb = y_va(firstInd:lastInd);
    
    Z1(:,1:mb_size) = V * x_batch(:,1:mb_size);
    X2(2:end,1:mb_size) = 1./(1+exp(-Z1(:,1:mb_size)));
    Z2 = W * X2;
    
    [~,pred] = max(Z2(:,1:mb_size));
    y_batch = zeros(K,mb_size);
    indd = sub2ind([K,mb_size],pred,1:mb_size);
    y_batch(indd) = 1;
    stats.score_va(:,firstInd:lastInd) = stats.score_va(:,firstInd:lastInd) + y_batch * alpha;
    err_va = err_va +sum(yb' ~= pred);
end
err_va = err_va / n2;
% fprintf('%d th iter error (valid): %f\n',N,err_va);

[~,pred_tr] = max(stats.score_tr);
stats.err_tr = [stats.err_tr mean(pred_tr~=y_tr')];

[~,pred_va] = max(stats.score_va);
stats.err_va = [stats.err_va mean(pred_va~=y_va')];

fprintf('\t\ttraining error: %f\n\t\tvalidation error: %f\n\n',stats.err_tr,stats.err_va);

