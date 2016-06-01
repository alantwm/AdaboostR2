function [hf]=adaboostr2(T,N)

n=size(T,1);
xo=T(:,1:end-1);yo=T(:,end);
w(:,1)=ones(1,n)/(n);
ind(:,1) = randsample(n,n,true,w) ;

for t=1:N
    xt=T(ind(:,t),1:end-1);
    yt=T(ind(:,t),end);
    
    % Building the Learner Model
    model=fitrsvm(xt,yt);
%     model=fitrtree(xt,yt);
    yp(:,t)=predict(model,xo);

    
    % Calculating Error
    err_i=abs(yp(:,t)-yo)/max(abs(yp(:,t)-yo));
    avg_err=sum(err_i.*w(:,t));
    beta(t)=avg_err/(1-avg_err);
    sum_beta(t)=sum(log(1./beta));
    
    %Updating Weights and Changing Indices for Data Sampling
    if t~=N
        w(:,t+1)=w(:,t).*beta(t).^(1-err_i);
        w(:,t+1)=w(:,t+1)/sum(w(:,t+1));
        ind(:,t+1) = randsample(n,n,true,w(:,t+1)) ;
    end

    figure(1);hold on;
    plot(t,avg_err,'o')
    
    
end

[yp_sorted,ind_yp] = sort(yp');
ln_beta=log(1./beta');
for i=1:n
    sort_beta=ln_beta(ind_yp(:,i));
    cumWeights = cumsum(sort_beta) ./ sum(sort_beta);    
    j = find(cumWeights<=.5,1,'last');
    hf(i)=yp(i,j+1);
end

% figure(2)
% plot(xo,hf,'o');hold on;
% plot(xo,yo,'x')