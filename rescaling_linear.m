function [ys_rescaled,B]=rescaling_linear(xt,yt,xs,ys)

type = 'function estimation';
[Yp,alpha,b,gam,sig2,model] = lssvm(xs,ys,type);
ys_at_t=simlssvm(model,xt);

X=[ones(size(ys_at_t,1),1) ys_at_t];
Y=yt;
B=X\Y;

ys_rescaled=ys*B(2)+B(1);