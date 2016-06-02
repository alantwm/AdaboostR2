function [output]=twostage_tradaboostr2(xt,yt,xs,ys,N)

    [ys,B]=rescaling_linear(xt,yt,xs,ys);

    n=size(ys,1);S=20;
    m=length(yt);  
    F=10;cv=cvpartition(m,'kfold',F);
beta=1;
    w(:,1)=ones(1,n+m)/(n+m);
    for t=1:2

        % Calculating Error
%         w_cv(:,t)=ones(1,sum(cv.training(1))+n)/(sum(cv.training(1))+n);
        for f=1:F
            ind_cv=[logical(ones(n,1));cv.training(f)];
            w_cv(:,t)=w(ind_cv,t);
            m=length(yt(cv.training(f),:));   

            T=[xs ys;xt(cv.training(f),:) yt(cv.training(f),:)];
            T_test=[xt(cv.test(f),:) yt(cv.test(f),:)];
            [yp_cv(f),ww]=adaboostr2p(T,T_test,N,w_cv(:,end),m);
            error_tmp(f)=mse(yp_cv(f)-T(cv.test(f),end));
        end
        error(t)=sqrt(mean(error_tmp));    

        T=[xs ys;xt yt];
        xo=[xs;xt];yo=[ys;yt];
        m=length(yt);    
        [zzzz,w(:,t)]=adaboostr2p(T,T,N,w(:,end),m);

        ind = randsample(n+m,n+m,true,w(:,t));
        T=T(ind,:);

        % Build Learner Model
        model=fitrtree(T(:,1:end-1),T(:,end));
        yp(:,t)=predict(model,xo);
        err_i=abs(yp(:,t)-yo)/max(abs(yp(:,t)-yo));
%         avg_err=sum(err_i.*w(:,t));

        % Update Weights
        beta=fminsearch(@wupdate,10);
        save_beta(t)=beta;
        w(:,t+1)=w(:,t).*beta.^(err_i);
        w(:,t+1)=w(:,t+1)/sum(w(:,t+1));       
        w
        t
        error
        save_beta
        
        output=1;


    end
    
    
        function loss=wupdate(beta)
        % Function to find beta
            if t~=N
                w(:,t+1)=w(:,t).*beta.^(err_i);
                w(:,t+1)=w(:,t+1)/sum(w(:,t+1));                     
                sum_wm=sum(w(n+1:n+m,t+1));
                theoretical_sum=m/(m+n)+t/(S-1)*(1-m/(n+m));
                loss=abs(sum_wm-theoretical_sum)
            end    

        end    
end

