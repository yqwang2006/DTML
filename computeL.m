function L = computeL(W,lastL,z,actfuncType,P,Q)

[d,N] = size(z);
L = cell(1,N);
for i = 1:N
    L{i} = sparse(d,N);
end
dev_z = devActfunc(z,actfuncType);
for i = 1:N
    
     L{i}(:,P(i,:)) = (W'*lastL{i}(:,P(i,:))) .* repmat(dev_z(:,i),1,size(P(i,:),2));
      for j = 1:size(P(i,:),2)
        L{P(i,j)}(:,i) = (W'*lastL{P(i,j)}(:,i)) .* dev_z(:,P(i,j));
      end 
     
     L{i}(:,Q(i,:)) = (W'*lastL{i}(:,Q(i,:))) .* repmat(dev_z(:,i),1,size(Q(i,:),2));
     for j = 1:size(Q(i,:),2)
        L{Q(i,j)}(:,i) = (W'*lastL{Q(i,j)}(:,i)) .* dev_z(:,Q(i,j));
     end 
end