clear
clc
close all

x=linspace(-5,5,200);
y=x.^3;
T=[x;y]';
N=20;
hf=adaboostr2(T,N);
model2=fitensemble(x',y','LSBoost',N,'Tree')
hf2=predict(model2,x');

figure(2);
plot(x,hf,'ko');hold on;
plot(x,hf2,'rx')
 plot(x,y,'.');