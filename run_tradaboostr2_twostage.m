close all
clear
clc

xs=linspace(-5,5)';
xt=linspace(-5,5,10)';
ys=10*xs.^2+15;
yt=5*xt.^3;
N=20;
[output]=twostage_tradaboostr2(xt,yt,xs,ys,N);
% 
% figure(11);subplot(2,1,1)
% plot(xs,ys,'r');hold on;
% plot(xt,yt,'b')
% 
% figure(11);subplot(2,1,2)
% plot(xs,ys_rescaled,'rx');hold on;
% plot(xt,yt,'b')
