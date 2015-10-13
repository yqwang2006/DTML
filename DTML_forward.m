function [y,z] = DTML_forward(x,W,b,opts)
z = W*x+repmat(b,1,size(x,2));    
y = actfunc(z,opts.actfuncType);
end

