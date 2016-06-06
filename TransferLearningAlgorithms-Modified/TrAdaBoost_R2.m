function [model,rmse] = TrAdaBoost_R2(train_x,train_f,source_x,source_f,test_x,test_f,S,N)
    F = 10; % cross validation folds
    % S is the number of steps allocated to the training process
    % N is the number of learners in an ensemble
%     if S<= 1 || N<=1
%         error('At least two iterations required per stage of TrAdaBoost.R2')
%     end
    [instances1,features] = size(train_x);
    instances2 = size(source_x,1);
    instances = instances1+instances2;
    distribution = (1/instances)*ones(instances,1);
    % Split training data into folds for cross validation
    permut = randperm(instances1);
    train_x = train_x(permut,:);
    train_f = train_f(permut);
    split_length = floor(instances1/F);
    counter = 0;
    for i = 1:F
        folds_x{i} = train_x(counter+1:counter+split_length,:);
        folds_f{i} = train_f(counter+1:counter+split_length);
        folds_indx{i} = counter+1:counter+split_length;
        counter = counter + split_length;
    end
    for i = 1:F
        CVfold(i).train_x = [];
        CVfold(i).train_f = [];
        CVfold(i).indx = [];
        for j = 1:F
            if i == j
                CVfold(i).test_x = folds_x{j};
                CVfold(i).test_f = folds_f{j};
            else
                CVfold(i).train_x = [CVfold(i).train_x;folds_x{j}];
                CVfold(i).train_f = [CVfold(i).train_f;folds_f{j}];
                CVfold(i).indx = [CVfold(i).indx folds_indx{j}];
            end
        end
        CVfold(i).m = length(CVfold(i).indx);
        CVfold(i).indx = [CVfold(i).indx instances1+1:instances];        
    end
    concat_x = [train_x;source_x];
    concat_f = [train_f;source_f];
    errors = zeros(1,S);
    train_in = zeros(instances,features);
    train_out = zeros(instances,1);
    for i = 1:S
        rmse_list = zeros(1,F);
        parfor j = 1:F
            [xxx,rmse_list(j)] = AdaBoost_R2m([CVfold(j).train_x;source_x],[CVfold(j).train_f;source_f],distribution(CVfold(j).indx),CVfold(j).test_x, ...
                CVfold(j).test_f,N,CVfold(j).m,instances2);
        end
        errors(i) = mean(rmse_list);
        [models{i},xxx] = AdaBoost_R2m(concat_x,concat_f,distribution,[],[],N,instances1,instances2);
%         for j = 1:instances
%             indx = RouletteWheelSelection(distribution);
%             train_in(j,:) = concat_x(indx,:);
%             train_out(j) = concat_f(indx);
%         end

        indx=randsample(instances,instances,true,distribution);
        train_out = concat_f(indx);train_in = concat_x(indx,:);
        
        hyp = fitrtree(train_in,train_out);
        y = predict(hyp,source_x);
        err = abs(source_f - y);
        norm_err = err/max(err);
        beta = fminbnd(@(x) betafunc(x,instances1,instances2,i,S,distribution,norm_err),0,1);
        distribution(instances1+1:instances) = distribution(instances1+1:instances).*beta.^norm_err;
        distribution = distribution/sum(distribution);
    end
    [xxx,indx] = min(errors);
    model = models{indx};
    rmse = 0;
    if ~isempty(test_x)
        mse = PredictErr(model,test_x,test_f);
        rmse= sqrt(mse);
    end  
end