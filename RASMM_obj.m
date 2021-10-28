% Author: Qian Chengde 
% E-Mail: qianchd(gmail)
% Date  : 2021-10-19
% Copyright 2021 Qian Chengde.
% File: RASMM_obj.m

% objection function

function value = RASMM_obj(M,X,y,W,p,q,n,s,s2,tau,gamma,C,K)
tem = 0;
for i =1:K-1
    tem = tem + sum(svd(reshape(M(i,:),p,q)));
end
B = X*M'*W;
B_2 = ones(n,K);
B_3 = ones(n,K)*(1-gamma);
B_4 = ones(n,K)*s2;
for i = 1:n
    B(i,y(i)) = -B(i,y(i));
    B_2(i,y(i)) = K-1;
    B_3(i,y(i)) = gamma;
    B_4(i,y(i)) = s;
end
B = (((B+B_2)>0).*(B+B_2)-((B+B_4)>0).*(B+B_4)).*B_3;
value = 0.5*norm(M,'fro')^2+ tau*tem + C*sum(sum(B));
end