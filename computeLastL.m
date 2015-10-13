function L = computeLastL(h,z,actfuncType,P,Q)
[d,N] = size(h);
L = cell(1,N);
for i = 1:N
    L{i} = sparse(d,N);
end
dev_z = devActfunc(z,actfuncType);

clear z;

for i = 1:N
    
    L{i}(:,P(i,:)) = (repmat(h(:,i),1,size(P(i,:),2))-h(:,P(i,:))) .* repmat(dev_z(:,i),1,size(P(i,:),2));
    for j = 1:size(P(i,:),2)
        L{P(i,j)}(:,i) = (h(:,P(i,j))-h(:,i)) .* dev_z(:,P(i,j));
    end 
    L{i}(:,Q(i,:)) = (repmat(h(:,i),1,size(Q(i,:),2))-h(:,Q(i,:))) .* repmat(dev_z(:,i),1,size(Q(i,:),2));
    for j = 1:size(Q(i,:),2)
        L{Q(i,j)}(:,i) = (h(:,Q(i,j))-h(:,i)) .* dev_z(:,Q(i,j));
    end 
end