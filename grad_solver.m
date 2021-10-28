% Author: Qian Chengde 
% E-Mail: qianchd(gmail)
% Date  : 2021-10-19
% Copyright 2021 Qian Chengde.
% File:grad_solver.m

% A solver for the SVM interation in ADMM.
function A = grad_solver(X,A_0,B,W,S,L,V,lbound,ubound,rho,gamma,C,disp_f,ep)
  g_ttl = 0;
  A = A_0;
  WTX = X'*((A-B)*W');
  LSW = (L+rho*S)'*W;
  XLSW = X*LSW;
  XWTXW = X*(WTX*W);
  grad = XWTXW + XLSW + (1+rho)*V;
  
  %disp('tunning alpha')
  
  alpha = 0.001*gamma*C/max(max(abs(grad)));
  alpha = tunning_alpha(alpha,grad,WTX,LSW,A,X,L,S,W,rho,V,lbound,ubound);
  
  i = 0;
  grad_earn = -10e+371;
  while i<25000 && grad_earn< -ep
    i = i+1;
    %disp('a')
    if mod(i,10)==0
        alpha = tunning_alpha(alpha,grad,WTX,LSW,A,X,L,S,W,rho,V,lbound,ubound);
    end
    %disp('b')
    delta_A = -alpha*grad;
    lower_list = delta_A+A < lbound;
    upper_list = delta_A+A > ubound;
    delta_A(lower_list) = lbound(lower_list)-A(lower_list);
    delta_A(upper_list) = ubound(upper_list)-A(upper_list);
    %disp('computing...')
    delta_WTX = X'*(delta_A*W');
    %disp({max(max(abs(grad))),max(max(abs(delta_A)))})
    
    if max(max(abs(delta_A)))==0
        break
    end
    
    grad_earn = norm(delta_WTX,'fro')^2 +2*(sum(sum(delta_WTX.*WTX))+sum(sum((X'*delta_A).*LSW)) + sum(sum(((1+rho)*V).*delta_A)));
    while grad_earn >=0 && alpha > 10e-20   
      %disp({'shrinkage alpha',alpha})
      %disp(norm(delta_WTX,'fro')^2)
      %disp(2*sum(sum(delta_WTX.*WTX)))
      %disp(sum(sum((X'*delta_A).*((L+rho*S)'*W))))
      %disp(sum(sum(((1+rho)*V).*delta_A)))
      alpha = alpha/2;
      delta_A = -alpha*grad;
      lower_list = delta_A+A < lbound;
      upper_list = delta_A+A > ubound;
      delta_A(lower_list) = lbound(lower_list)-A(lower_list);
      delta_A(upper_list) = ubound(upper_list)-A(upper_list);
      delta_WTX = X'*(delta_A*W');
      grad_earn = norm(delta_WTX,'fro')^2 +2*(sum(sum(delta_WTX.*WTX))+sum(sum((X'*delta_A).*LSW)) + sum(sum(((1+rho)*V).*delta_A)));
    end
    
    g_ttl = g_ttl+grad_earn;
    %disp({'time',i})
    %disp({'alpha,',alpha})
    if mod(i,disp_f)==0
        disp({i,max(max(abs(delta_A))),grad_earn,g_ttl})
    end
    A = A + delta_A;
    WTX = WTX + delta_WTX;
    %grad_old = grad;
    grad = X*(delta_WTX*W) + grad;
    
    %grad = 0.95*grad + 0.05*grad_old;
  end
end