function mse = PredictErr(model,test_x,test_f)
    n = length(model.w);
    instances = size(test_x,1);
    sum_weights = 0.5*sum(model.w);
    mse = 0;
    for i = 1:instances
        prediction = zeros(1,n);
        for j = 1:n
            prediction(j) = predict(model.M{j},test_x(i,:));
        end
        [prediction,indx] = sort(prediction);
        weights = model.w(indx);
        partial_sum = 0;
        for j = 1:n
            partial_sum = partial_sum + weights(j);
            if partial_sum >= sum_weights
                break;
            end
        end
        mse = mse + (prediction(j)-test_f(i))^2;
    end
end