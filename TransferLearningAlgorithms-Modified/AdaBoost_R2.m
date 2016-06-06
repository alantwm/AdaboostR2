% Returns ensemble as a cell array model.M of n learnt models comprising
% the overall hypothesis, and a numeric array model.w of model weights
function [model,rmse] = AdaBoost_R2(train_x,train_f,test_x,test_f,n)
%     clc
    if size(train_f,2) > 1 || size(test_f,2) > 1
        error('Maximum of 1 output label allowed')
    end
%     n = 1; % Number of hypotheses/learners
    [instances,features] = size(train_x);
    distribution = (1/instances)*ones(instances,1);
    model.w = [];
    counter = 1;
    if n == 1
        model.M{1} = fitrtree(train_x,train_f);
        model.w = 1;
    else
        train_in = zeros(instances,features);
        train_out = zeros(instances,1);
        for i = 1:n            
%             for j = 1:instances
%                 indx = RouletteWheelSelection(distribution);
% %                 save_indx(j)=indx;
%                 train_in(j,:) = train_x(indx,:);
%                 train_out(j) = train_f(indx);
%             end

            indx=randsample(instances,instances,true,distribution);
            train_out = train_f(indx);train_in = train_x(indx,:);

            hyp = fitrtree(train_in,train_out);
            y = predict(hyp,train_x);
            err = abs(train_f - y);
            norm_err = err/max(err);        
            epsilon = norm_err'*distribution;
            if epsilon >= 0.5
                n=i-1;
                break;
            end
            beta = epsilon/(1-epsilon);
            distribution = distribution.*beta.^(1-norm_err);
            distribution = distribution/sum(distribution);
            model.M{counter} = hyp;
            model.w = [model.w log(1/beta)];
            counter=counter+1;
        end
    end
%     model.M{1}
    rmse = 0;
    if nargin > 2
        mse = PredictErr(model,test_x,test_f);
        rmse= sqrt(mse);
    end    
end