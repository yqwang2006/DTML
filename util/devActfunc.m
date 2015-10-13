function [ y ] = devActfunc( x, actfuncType )
%ACTFUNC 此处显示有关此函数的摘要
%   此处显示详细说明
switch actfuncType
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        y = x .* (ones(size(x))-x);
    case {'tanh'}
        y = 1-tanh(x).^2;
    case {'linear'}
        y = ones(size(x));
end

end