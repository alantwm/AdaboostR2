function [hf,wt]=adaboostr2p(T,T_test,N,w,m)
% Adaboost.R2' for Two-Stage TrAdaboost.R2.
n=size(T,1);
x_test=T_test(:,1:end-1);y_test=T_test(:,end);
xo=T(:,1:end-1);yo=T(:,end);
ind(:,1) = randsample(n,n,true,w);
tmp=w(1:(n-m),1);
for t=1:N
    xt=T(ind(:,t),1:end-1);
    yt=T(ind(:,t),end);
    
        % Building the Learner Model
%         model=fitrsvm(xt,yt);
        model=fitrtree(xt,yt);
        yp(:,t)=predict(model,xo);
        yp_test(:,t)=predict(model,x_test);
    
    % Calculating Error
    err_i=abs(yp(:,t)-yo)/max(abs(yp(:,t)-yo));
    avg_err=sum(err_i.*w(:,t));
    
    err_test_i=abs(yp_test(:,t)-yo)/max(abs(yp_test(:,t)-yo));
    avg_err_test=sum(err_test_i.*w(:,t));
    
    beta(t)=avg_err/(1-avg_err); 
    beta_test(t)=avg_err_test/(1-avg_err_test); 
    %Updating Weights and Changing Indices for Data Sampling
    if t~=N
        w(:,t+1)=w(:,t).*beta(t).^(1-err_i);
        w(:,t+1)=w(:,t+1)/sum(w(:,t+1));
        w(1:(n-m),t+1)=tmp;
        ind(:,t+1) = randsample(n,n,true,w(:,t+1)) ;
    end

%     figure(1);hold on;
%     plot(t,avg_err,'o')
    
    
end

[yp_test_sorted,ind_yp] = sort(yp_test');
ln_beta=log(1./beta_test');
for i=1:length(y_test)
    sort_beta=ln_beta(ind_yp(:,1));
    cumWeights = cumsum(sort_beta) ./ sum(sort_beta);    
    j = find(cumWeights>=.5,1,'first');
    hf(i)=yp_test(i,j);
    wt(i,1)=w(i,j);
end

% figure(2)
% plot(xo,hf,'o');hold on;
% plot(xo,yo,'x')