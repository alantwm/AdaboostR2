function rmse = GenErr(model_s,model_t,b,test_x,test_f)
    prediction = treeval(model_t,test_x);
    error=test_f-prediction;
    rmse = sqrt(error'*error)
    
    prediction = b(1)*treeval(model_s,test_x) + b(2)*treeval(model_t,test_x);
    error=test_f-prediction;
    rmse = sqrt(error'*error);
end