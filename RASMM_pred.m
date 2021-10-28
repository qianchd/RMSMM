% Author: Qian Chengde 
% E-Mail: qianchd(gmail)
% Date  : 2021-10-19
% Copyright 2021 Qian Chengde.
% File: RASMM_pred.m

% prediction

function [y_pred] = RASMM_pred(X,M,K)
W = [ones(K-1,1)/sqrt(K-1), -((1+sqrt(K))/(K-1)^1.5)*ones(K-1,K-1) + sqrt(K/(K-1))*eye(K-1)];
[~,y_pred] = max(W'*M*X');
y_pred = y_pred';
end