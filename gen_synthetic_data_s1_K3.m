% Author: Qian Chengde 
% E-Mail: qianchd(gmail)
% Date  : 2021-10-19
% Copyright 2021 Qian Chengde.
% File: gen_synthetic_data_s1_K3.m

% gen synthetic data for simulaiton 1 with K = 3
K = 3;
rank = 5;
n = 21000;
v = rank*K;   % rank * K < p
p = 50;
q = 50;
d = q/rank;
%sigma = 0.05;
%V = orth(rand(p,v));
%V2= orth(rand(p,v));

X = zeros(n,p*q);
G = zeros(K,p*q);
for i = 1:K
    V = orth(rand(p,rank));
    V2= orth(rand(q,rank));
    G(i,:) = reshape(V*V2',1,p*q);
end
%tem = 0.1*M(y(1),:) + 0.9*M(y(2),:);
%M(y(1),:) = 0.9*M(y(1),:) + 0.1*M(y(2),:);
%M(y(2),:) = tem;

y = ceil(rand(n,1)*K);
for i = 1:n
    X(i,:) = G(y(i),:) + normrnd(0,sigma,1,p*q);
end

r = X*G';
[~,l] = max(r');
sum(y==l')/n