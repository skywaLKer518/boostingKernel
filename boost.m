DEVICE = 1;
NUM_HIDDEN = 5;
NUM_ITER = 5;
BATCH_SIZE = 50;
LEARNING_RATE = 1;
% NUM_MODEL = 100;
fprintf('input: NUM_MODEL, NUM_HIDDEN, NUM_ITER, LEARNING_RATE\n')
keyboard
%%

fprintf('loading data\n')
load mnist_60k
ind = randperm(size(labelTr,1));
no_tr = 50000;
no_va = 10000;

dataTr = dataTr / 256;
dataTr = [ones(1,size(dataTr,2)); dataTr]; % append 1 for each instance
x_tr = single(dataTr(:,ind(1:no_tr)));
y_tr = labelTr(ind(1:no_tr))+1; % mnist 1 to 10
x_va = single(dataTr(:,ind(1+no_tr:end)));
y_va = labelTr(ind(1+no_tr:end))+1;

clearvars labelTr dataTr;

%% hyper-parameters etc
hypers.n_hid = NUM_HIDDEN;
hypers.n_train = length(y_tr);
hypers.dim = size(x_tr,1);
hypers.no_classes = max(y_tr);
hypers.nIters = NUM_ITER;
hypers.batch_size = BATCH_SIZE;
hypers.learning_rate = LEARNING_RATE;
hypers.device = DEVICE;
%%
stats.score_tr = zeros(hypers.no_classes,no_tr);
stats.score_va = zeros(hypers.no_classes,no_va);
stats.err_tr = [];
stats.err_va = [];

M = cell(NUM_MODEL,1);
weight = ones(1,no_tr) * 1/no_tr;
% weight = normrnd(0.5,0.01,1,no_tr);
% weight = weight - min(weight);
% weight = weight / sum(weight);

% weight(0.95*no_tr:end) = 100;
% weight = weight / sum(weight);
% keyboard
%%
tic
for i = 1:NUM_MODEL
    fprintf('%d (%d,%d,%f)-- ',i,NUM_ITER,NUM_HIDDEN,LEARNING_RATE);
    model= train_net(x_tr,y_tr,weight,hypers);
    M{i} = model;
    [weight] = updateWeigts(weight,y_tr,model.pred,model.alpha);
    stats = eval_model(x_tr,y_tr,x_va,y_va,i,model,stats); 
    
end
total_time = toc;

name = ['output/res_iter'  num2str(NUM_ITER) '_lr' num2str(LEARNING_RATE) ...
    '_N' num2str(NUM_MODEL) '_hid' num2str(NUM_HIDDEN)];
clearvars  x_tr x_va y_tr y_va no_tr no_va
save(name)