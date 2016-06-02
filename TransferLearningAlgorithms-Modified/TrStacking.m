% Test of transfer stacking
function rmse = TrStacking(train_x,train_f,source_x,source_f,test_x,test_f)
    F = 5; % cross validation folds
    [instances1,xxx] = size(train_x);
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
    model_s = treefit(source_x,source_f);
    model_t = treefit(train_x,train_f);
    FeatureMat = [];
    Output = [];
    for i = 1:F
        CVfold(i).train_x = [];
        CVfold(i).train_f = [];
        for j = 1:F
            if i == j
                CVfold(i).test_x = folds_x{j};
                CVfold(i).test_f = folds_f{j};
            else
                CVfold(i).train_x = [CVfold(i).train_x;folds_x{j}];
                CVfold(i).train_f = [CVfold(i).train_f;folds_f{j}];
            end
        end
        model = treefit(CVfold(i).train_x,CVfold(i).train_f);
        temp = [treeval(model_s,CVfold(i).test_x) treeval(model,CVfold(i).test_x)];
        FeatureMat = [FeatureMat;temp];
        Output = [Output;CVfold(i).test_f];
    end   
    b = mvregress(FeatureMat,Output);
    rmse = GenErr(model_s,model_t,b,test_x,test_f);
end