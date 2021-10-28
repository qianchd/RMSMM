% Author: Qian Chengde 
% E-Mail: qianchd(gmail)
% Date  : 2021-10-19
% Copyright 2021 Qian Chengde.
% File: RASMM_admm.m

% The admm solver for RASMM

function [M,S_head,A,W,M_SMM,M_2,accu_SMM,accu_2,accu_final] = RASMM_admm(X,y,X_test,y_test,p,q,K,C,M_0,gamma,s,s2,tau,rho,disp_f,issmm_only,issmm)
  % X n rows, every row is a obs
  % p and q: dimension of obs
  % C the SVM parameter
  % K number of classes 
  % m_0 create the initial M
  % s the parameter in the truncated loss
  % tau the parameter in the trace norm 0<tau<0.5
  % rho the paraameter in Admm lagrangian multipler
  % disp_f: control the disp procedure
  % issmm_only: if true, the DCA loop will stop at iterate 1; be 0 commonly.
  % issmm: if true, the first iteration in the DCA loop is the SMM solution.
   


% d.c. algorithm
  % M K-1 * pq; X n * pq; W K-1 * K;
  n = length(y);
  C = C/n;
  
  W = [ones(K-1,1)/sqrt(K-1), -((1+sqrt(K))/(K-1)^1.5)*ones(K-1,K-1) + sqrt(K/(K-1))*eye(K-1)];
  
  
  M_2=M_0;
  % initialize for d.c.
  M = M_0;
  M_dc_old = M+1;
  S_head = M;
  L_head = M;
  
  % for dc convergence
  %dif_norm = 0;
  %conv_t = 0;
  
  V = ones(n,K);
  lbound = ones(n,K)*(gamma-1)*C;
  ubound = zeros(n,K);
  for i = 1:n
    V(i,y(i)) = 1-K;
    ubound(i,y(i)) = gamma*C;
    lbound(i,y(i)) = 0;
  end
  
  A = (lbound+ubound)/2;
  
  obj_v = inf;
  ep_dc= 1e-7*C; 
  ep_admm = 1e-5*C;
  ep=ep_dc;
  ep_a = ep_admm;
  
  for t_dc = 1:100
     %if norm(M_dc_old-M,'fro')< 0.1
     %    conv_t = conv_t+1;
     %else
     %    conv_t = 0;
     %end
     %disp({'t_dc',t_dc,'diff F-norm of M',norm(M_dc_old-M,'fro'),'conv_time',conv_t})
     
     
     M_dc_old = M;
     
     B = X*M'*W+s2;
     B_2 = ones(n,K)*C*(gamma-1);
     for i = 1:n 
      B(i,y(i)) = s+s2-B(i,y(i));
      B_2(i,y(i)) = B_2(i,y(i)) + C;
     end
     B = (B>0).*B_2;
     
     if t_dc == 1 && issmm == 1
         B = zeros(n,K);
     end
     
     %disp('starting ADMM')
     
     admm_conv = 0;
     M_admm_old = zeros(K-1,p*q);
     if tau == 0
         S_head = zeros(K-1,p*q);
         L_head = zeros(K-1,p*q);
     end
     for a = 1:1000
         A = grad_solver(X,A,B,W,S_head,L_head,V,lbound,ubound,rho,gamma,C,disp_f,ep);
         M = (L_head+rho*S_head+ W*(A-B)'*X)/(1+rho);
         %disp(norm(M_admm_old-M,'fro')/norm(M_admm_old+M,'fro'))
         
         if tau == 0
             break
         end
         
         if norm(M_admm_old-M,'fro')/norm(M_admm_old+M,'fro')< ep_a
             admm_conv = admm_conv+1;
             %disp(admm_conv)
         end
         if admm_conv >4
             %disp({'admm break,admm steps:',a})
             break;
         end
         M_admm_old = M;
     
         tem_M = rho*M-L_head;
         for i=1:K-1
             S_head(i,:) =reshape(svt(reshape(tem_M(i,:),[p,q]),tau),[1,p*q])/rho;
         end
         L_head = L_head - rho*(M - S_head);
     end
     %disp({'t_dc',t_dc,'diff F-norm of M',norm(M_dc_old-M,'fro')})
     y_pred = RASMM_pred(X_test,M,K);
     obj_tem = RASMM_obj(M,X,y,W,p,q,n,s,s2,tau,gamma,C,K);
     accu_rate = sum(y_pred==y_test)/length(y_test);
     %if mod(t_dc,3)==1
     %     disp({'t_dc',t_dc,'diff F-norm of M',norm(M_dc_old-M,'fro')/norm(M_dc_old+M,'fro'),'accu rate',accu_rate,'loss on test set',RASMM_loss(M,X_test,y_test,W,s,s2,gamma,K)})
     %end
     %disp({'loss on test set',RASMM_loss(M,X_test,y_test,W,s,s2,gamma,K),'C',C})
     if t_dc == 1
         M_SMM = M;
         accu_SMM = accu_rate;
         accu_2 = -1;
         accu_final = -1;
         if issmm_only == 1
             break;
         end
     end
     if t_dc == 2
         M_2 = M;
         accu_2 = accu_rate;
     end
     if accu_rate < accu_2 - 0.02
         %break
     end
     if norm(M_dc_old-M,'fro')/norm(M_dc_old+M,'fro')< 4e-4
         break;
     end
     if obj_tem >= obj_v
         %disp({'dca break because obj increases',obj_tem - obj_v,obj_tem})
         M = M_dc_old;
         ep = ep/1000;
         ep_a = ep_a/10;
         %break
     else
         obj_v = obj_tem;
         accu_final = accu_rate;
         ep=ep_dc;
         ep_a = ep_admm;
     end
     if ep <ep_dc*10e-9
         break;
     end
  end
end