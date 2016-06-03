function [prob] = ordersample2(K,numbersPicked,N,current)
% ORDERSAMPLE  Implements the prior. This is the probabilistic equivalent of 
%              randomly sampling without replacement, and then sorting the sequence
%              This generates the probabilities for selected each element in that way
%              This function additionally only returns the distribution over the remaining elements
%
% Gunnar Atli Sigurdsson & Xinlei Chen 2015
% Carnegie Mellon University

    prob = zeros(1,N);
    not = 1;
    for i=current:N-(K-numbersPicked)+1
        s = (K-numbersPicked)/(N-i+1);
        prob(i) = s*not;
        not = not * (1-s);
    end
    %prob = prob(current:N);
end

