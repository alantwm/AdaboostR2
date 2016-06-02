% Returns ensemble as a cell array model.M of n learnt models comprising
% the overall hypothesis, and a numeric array model.w of model weights
function [model,rmse] = AdaBoost_R2m(train_x,train_f,distribution,test_x,test_f,N,m,n)
%     N is the number of learners, m is the number of target instances, n
%     is the number of sources instances
%     clc
    distribution = distribution/sum(distribution);
    if size(train_f,2) > 1 || size(test_f,2) > 1
        error('Maximum of 1 output label allowed')
    end
    [instances,features] = size(train_x);
    if instances ~= n+m || instances ~= length(distribution)
        error('Number of instances does not match sum of source and target')
    end
    model.w = [];
    if N == 1
        model.M{1} = treefit(train_x,train_f);
    else
        train_in = zeros(instances,features);
        train_out = zeros(instances,1);
        counter = 1;
        for i = 1:N            
%             for j = 1:instances
%                 indx = RouletteWheelSelection(distribution);
%                 save_indx(j)=indx;
%                 train_in(j,:) = train_x(indx,:);
%                 train_out(j) = train_f(indx);
%             end
            indx=randsample(instances,instances,true,distribution);
            train_out = train_f(indx);train_in = train_x(indx,:);
            hyp = fitrtree(train_in,train_out);
            y = predict(hyp,train_x(1:m,:));
            err = abs(train_f(1:m) - y);
            norm_err = err/max(err);        
            epsilon = norm_err'*distribution(1:m);
            if epsilon >= 0.5
                n=i-1;
                break;
            end
            beta = epsilon/(1-epsilon);
            distribution(1:m) = distribution(1:m).*beta.^(1-norm_err);
            distribution = distribution/sum(distribution);
            model.M{counter} = hyp;
            model.w = [model.w log(1/beta)];
            counter=counter+1;
        end
    end
%     model.M{1}
    rmse = 0;
    if ~isempty(test_x)
        mse = PredictErr(model,test_x,test_f);
        rmse= sqrt(mse);
    end    
end