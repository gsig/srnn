function [c] = rnn_div(X,k)
% kmeans++-like init

N = size(X,1);
if N <= k
    c = 1:N;
    return;
end
c = randi(N,1);
Xnorm = sqrt(sum(X.^2,2));
for i=2:k
    D = (X*X(c,:)')./(Xnorm*Xnorm(c,:)'); %cosine distance
    D = D-min(D(:));
    D = D./max(D(:));
    d = min(D,[],2);
    w = d.^2;
    w(c) = 0;
    w = w/sum(w);
    newc = sum(rand(1) > cumsum(w))+1;
    c = [c newc];
end

end
