clc
clear
load EPDS_10CVpartitionedUC2_100



target_x=EPDS_UC2b(indx2,1:5);
target_f=EPDS_UC2b(indx2,6);

source_x=EPDS_UC2a(:,1:5);
source_f=EPDS_UC2a(:,6);

[source_f,B]=rescaling_linear(target_x,target_f,source_x,source_f);

for i=1:10
    cv=cvpartition(length(target_f),'kfold',10);

    train_x=target_x(cv.training(i),:);
    train_f=target_f(cv.training(i),:);

    test_x=target_x(cv.test(i),:);
    test_f=target_f(cv.test(i),:);



    [xxx,rmse_single(i)] = AdaBoost_R2(train_x,train_f,test_x,test_f,30);
    rmse_single
    [xxx,rmse_naive(i)] = TrAdaBoost_R2(train_x,train_f,source_x,source_f,test_x,test_f,1,30);
    rmse_naive
    [xxx,rmse_transfer(i)] = TrAdaBoost_R2(train_x,train_f,source_x,source_f,test_x,test_f,30,30);
    rmse_transfer 
    % Results = 0.6030, 0.5624
end

avg_rmse_single=mean(rmse_single)
rmse_naive=mean(rmse_naive)