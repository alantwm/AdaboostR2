function [index] =  RouletteWheelSelection(arrayInput)

len = length(arrayInput);

% if input is one element then just return rightaway
if len ==1
    index =1;
    return;
end

temp = 0;
cumProb = zeros(1,len);

% normalise inputs to be a well defined distribution
arrayInput = arrayInput/sum(arrayInput);

for i= 1:len
    cumProb(i) = temp + arrayInput(i);
    temp = cumProb(i);
end

i_rand = rand;

for i = 1:len
    if i_rand<=cumProb(i)
        index = i;
        return
    end
end

t=1;


