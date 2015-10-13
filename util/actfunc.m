function [ y ] = actfunc( x, actfuncType )
%ACTFUNC 此处显示有关此函数的摘要
%   此处显示详细说明
switch actfuncType
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        y = 1 ./ (1 + exp(-x));
    case {'sin','sine'}
        %%%%%%%% Sine
        y = sin(x);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        y = double(hardlim(x));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        y = tribas(x);
    case {'radbas'}
        %%%%%%%% Radial basis function
        y = radbas(x);
    case {'tanh'}
        y = tanh(x);
    case {'linear'}
        y = x;
end

end

