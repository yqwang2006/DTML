function [Jcost,Sc,Sb,Dts,normValue] = getJcost(Sc,Sb,Dts,net,opts)
Jcost = Sc - opts.alpha * Sb + opts.beta * Dts;
normValue = 0;
for m = 1:opts.M
    normValue = normValue + (norm(net.layer{m}.W,'fro')^2 + norm(net.layer{m}.b,2)^2);
end
Jcost = Jcost + opts.gamma * normValue;