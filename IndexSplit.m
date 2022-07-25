% function file: Index Split

% Input: parent indices I_tau for parent node tau with children alpha, beta
% Output: I_alpha, I_beta index vectors 

function [I_alpha, I_beta] = IndexSplit(I_tau)

n_tau = length(I_tau);
n_alpha = floor(n_tau/2);

I_alpha = I_tau(1:n_alpha);
I_beta = I_tau(n_alpha+1:n_tau);

end