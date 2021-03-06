function Dts = getDts(Xs,Xt)
[d,Nt] = size(Xt);
[d,Ns] = size(Xs);
Dts = norm((sum(Xt,2)./Nt-sum(Xs,2)./Ns).^2,2)^2;