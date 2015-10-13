function net = DTML_train(Xs,Labels,Xt,opts)
[d, N] = size(Xs);
%%%%%% Initial W and b %%%%%%%%%%
inputD = d;
for m = 1:opts.M %逐层前向传播
    hidNum = opts.hidNum(m);
    net.layer{m}.W = eye(hidNum,inputD);
    net.layer{m}.b = zeros(hidNum,1);
    inputD = hidNum;   
end
%%%%%% End Initial %%%%%%%%%%%%%

%%%%%% Get sparse P and Q %%%%%%%%%%%%
[P,Q] = getPQ(Xs,Labels,opts.k1,opts.k2,opts.distType);
%%%%%% End getting P and Q %%%%%
Sc = cell(1,opts.M);
Sb = cell(1,opts.M);
Dts = cell(1,opts.M);


%%%%%%%%%%%%%% Training %%%%%%%%%%%%%%%%%%%%%
lastJcost = inf;
for k = 1:opts.T
    inputLayer_s = Xs;
    inputLayer_t = Xt;
    hs = cell(1,opts.M);
    ht = cell(1,opts.M);
    zs = cell(1,opts.M);
    zt = cell(1,opts.M);
    dJdW = cell(1,opts.M);
    dJdB = cell(1,opts.M);
    
    for m = 1:opts.M %逐层前向传播
        [hs{m},zs{m}] =  DTML_forward(inputLayer_s,net.layer{m}.W,net.layer{m}.b,opts);
        [ht{m},zt{m}] = DTML_forward(inputLayer_t,net.layer{m}.W,net.layer{m}.b,opts);
        inputLayer_s = hs{m};
        inputLayer_t = ht{m};
    end
    clear inputLayer_s inputLayer_t;
    [Sc{opts.M},Sb{opts.M}] = getSbSc(P,Q,hs{opts.M},opts.k1,opts.k2,opts.distType);
    Dts{opts.M} = getDts(hs{opts.M},ht{opts.M});
    for m = opts.M:-1:1
        if m == opts.M
            L = computeLastL(hs{m},zs{m},opts.actfuncType,P,Q);
            Lt = computeLastLsLt(ht{m},hs{m},zt{m},opts.actfuncType);
            Ls = computeLastLsLt(hs{m},ht{m},zs{m},opts.actfuncType);            
        else
            L = computeL(net.layer{m+1}.W,lastL,zs{m},opts.actfuncType,P,Q);
            Lt = computeLsLt(net.layer{m+1}.W,lastLt,zt{m},opts.actfuncType);
            Ls = computeLsLt(net.layer{m+1}.W,lastLs,zs{m},opts.actfuncType);
        end
        if m == 1
            [dJdW{m},dJdB{m}] = getDelta(P,Q,L,Lt,Ls,net.layer{m}.W,net.layer{m}.b,Xs,Xt,opts);
        else
            [dJdW{m},dJdB{m}] = getDelta(P,Q,L,Lt,Ls,net.layer{m}.W,net.layer{m}.b,hs{m-1},ht{m-1},opts);
        end
        lastL = L;
        lastLs = Ls;
        lastLt = Lt;
    end

    
    for m = 1:opts.M
        net.layer{m}.W = net.layer{m}.W - opts.lr * dJdW{m};
        net.layer{m}.b = net.layer{m}.b - opts.lr * dJdB{m};
    end
    
    opts.lr = opts.lr * opts.lr_decay;
    
    [Jcost,Sc1,Sb1,Dts1,normValue] = getJcost(Sc{opts.M},Sb{opts.M},Dts{opts.M},net,opts);
    if abs(Jcost - lastJcost) < opts.epsilon
        break;
    end
    lastJcost = Jcost;
    fprintf('%d/%d, Jcost = %f, Sc = %f, Sb = %f, Dts = %f, norm = %f\n',k,opts.T,Jcost,Sc1,Sb1,Dts1,normValue);
end

