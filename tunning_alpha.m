% Author: Qian Chengde 
% E-Mail: qianchd(gmail)
% Date  : 2021-10-19
% Copyright 2021 Qian Chengde.
% File: tunning_alpha.m

% Grid search for learning rate

function alpha = tunning_alpha(alpha,grad,WTX,LSW,A,X,L,S,W,rho,V,lbound,ubound)
  delta_A = -alpha*grad;
  lower_list = delta_A < lbound-A;
  upper_list = delta_A > ubound-A;
  delta_A(lower_list) = lbound(lower_list)-A(lower_list);
  delta_A(upper_list) = ubound(upper_list)-A(upper_list);
  delta_WTX = X'*(delta_A*W');
  
  earning_new = norm(delta_WTX,'fro')^2 +2*(sum(sum(delta_WTX.*WTX))+sum(sum((X'*delta_A).*LSW)) + sum(sum(((1+rho)*V).*delta_A)));
  earning = -1;
  while earning<0
    alpha = 5*alpha;
    delta_A = -alpha*grad;
    lower_list = delta_A < lbound-A;
    upper_list = delta_A > ubound-A;
    delta_A(lower_list) = lbound(lower_list)-A(lower_list);
    delta_A(upper_list) = ubound(upper_list)-A(upper_list);
    delta_WTX = X'*(delta_A*W');
    
    earning_old = earning_new;
    earning_new = norm(delta_WTX,'fro')^2 +2*(sum(sum(delta_WTX.*WTX))+sum(sum((X'*delta_A).*LSW)) + sum(sum(((1+rho)*V).*delta_A)));
    earning = earning_new - earning_old ;
    %disp({'earning_new:',earning_new,'earning_old',earning_old})
    %disp({'alpha_u',alpha})
  end
  alpha = alpha/5;
  
  earning_new = earning_old;
  earning = -1;
  while earning<0
    alpha = alpha/2;
    delta_A = -alpha*grad;
    lower_list = delta_A < lbound-A;
    upper_list = delta_A > ubound-A;
    delta_A(lower_list) = lbound(lower_list)-A(lower_list);
    delta_A(upper_list) = ubound(upper_list)-A(upper_list);
    delta_WTX = X'*(delta_A*W');
    
    earning_old = earning_new;
    earning_new = norm(delta_WTX,'fro')^2 +2*(sum(sum(delta_WTX.*WTX))+sum(sum((X'*delta_A).*LSW)) + sum(sum(((1+rho)*V).*delta_A)));
    earning = earning_new - earning_old ;
    %disp({'alpha_d',alpha})
  end
  alpha =2*alpha;
end