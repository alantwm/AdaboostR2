clear
clc
close all

m = 25;
n = 1000;
test_size = 10000;

train_x = rand(m,10);
test_x = rand(test_size,10);
source_x = rand(n,10);

train_f = Friedman(train_x,0);
test_f = Friedman(test_x,0);
source_f = Friedman(source_x,1);

maxf = max([train_f;test_f;source_f]);
train_f = train_f/maxf;
test_f = test_f/maxf;
source_f = source_f/maxf;

% [source_f,B]=rescaling_linear(train_x,train_f,source_x,source_f);

%%%%%%%%%%%%%%%%%%%%%%%% Transfer AdaBoost %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% [xxx,rmse_single] = AdaBoost_R2(train_x,train_f,test_x,test_f,30);
% rmse_single
% [xxx,rmse_naive] = TrAdaBoost_R2(train_x,train_f,source_x,source_f,test_x,test_f,1,30);
% rmse_naive
[xxx,rmse_transfer] = TrAdaBoost_R2(train_x,train_f,source_x,source_f,test_x,test_f,30,30);
rmse_transfer

%%%%%%%%%%%%%%%%%%%%%%%% Transfer Stacking %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  rmse_transfer = TrStacking(train_x,train_f,source_x,source_f,test_x,test_f);
%  rmse_transfer