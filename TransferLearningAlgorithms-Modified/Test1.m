clc
load 'VEE_UC1a.txt'
data = VEE_UC1a;

data = 10*rand(250,5);
for i = 1:250
    data(i,6) = (data(i,1:5)-5)*(data(i,1:5)-5)';
end

instances = size(data,1);
data = data(randperm(instances),:);

train = floor(0.7*instances);
test = instances-train;
train_x = data(1:train,1:5);
train_f = data(1:train,6);
test_x = data(train+1:train+test,1:5);
test_f = data(train+1:train+test,6);
[model,generalization_error] = AdaBoost_R2(train_x,train_f,test_x,test_f,30)
% [model,generalization_error] = AdaBoost_R2(train_x,train_f,test_x,test_f,1)
% [model,generalization     _error] = AdaBoost_R2(train_x,train_f,[],[])