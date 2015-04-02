function ind = sample2ind(sample,n)
% given sample 1 by P
[~,b] = find(sample);
ind = zeros(1,n);

p = 1;
for j = 1:length(b)
    ind(p:p+sample(b(j))-1) = b(j);
    p=p+sample(b(j));
end
