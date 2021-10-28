% Author: Qian Chengde 
% E-Mail: qianchd(gmail)
% Date  : 2021-10-19
% Copyright 2021 Qian Chengde.
% File: svt.m

% SVD truncation

function D_t = svt(X,tau)
  [U,S,V] = svd(X);
  S = (S>tau).*(S-tau);
  D_t = U*S*V';
end