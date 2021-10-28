% Author: Qian Chengde 
% E-Mail: qianchd(gmail)
% Date  : 2021-10-19
% Copyright 2021 Qian Chengde.
% File: RASMM_pd.m

% The primal-dual solver for RASMM

function [M,S_head,A,W,M_SMM,M_2,accu_SMM,accu_2,accu_final] = RASMM_pd(X,y,X_test,y_test,p,q,K,C,M_0,gamma,s,s2,tau,rho,disp_f,issmm_only,issmm)
  % X: n rows, every row is a obs
  % p and q: dimension of obs
  % C: the SVM parameter
  % K: number of classes 
  % M_0: create the initial M
  % s and s2: the parameter in the truncated loss
  % tau: the parameter in the trace norm 0<tau<0.5
  % rho: the parameter in Admm lagrangian multipler
  % disp_f: control the disp procedure
  % issmm_only: if true, the DCA loop will stop at iterate 1; be 0 commonly.
  % issmm: if true, the first iteration in the DCA loop is the SMM solution.
  A = 0;
  S_head = 0;
  M_2 = 0;
  accu_2=0;
  M_SMM = 0;
  accu_SMM = 0;


% d.c. algorithm
  % M K-1 * pq; X n * pq; W K-1 * K;
  % initialize for d.c.
  n = length(y);
  lambda = 1/C;
  
  W = [ones(K-1,1)/sqrt(K-1), -((1+sqrt(K))/(K-1)^1.5)*ones(K-1,K-1) + sqrt(K/(K-1))*eye(K-1)];
  M = M_0;
  M_head = M;
  

  V = (1-gamma)*ones(K,n)/(2*lambda*n);
  Weight = (1-gamma)*ones(K,n)/(2*lambda*n);
  mu = (1-gamma)*ones(K,n)/(2*lambda*n);
  for i = 1:n
    V(y(i),i) = -gamma/(2*lambda*n);
    Weight(y(i),i) = gamma/(2*lambda*n);
    mu(y(i),i) = -gamma*(K-1)/(2*lambda*n);
  end
  mu = reshape(mu,n*K,1);
  d0 = reshape(W*V*X,(K-1)*p*q,1);
  v = reshape(Weight,n*K,1);
  %A_head = kron(X,W');
  %for i = 1:n*K
      %A_head(i,:) = A_head(i,:)*v(i);
  %end
  
  obj_v = inf;
  ep_dc= 5e-4; 
  ep=ep_dc;
  
  Y = zeros(K*n+1,1);

  
  for t_dc = 1:10
     %if norm(M_dc_old-M,'fro')< 0.1
     %    conv_t = conv_t+1;
     %else
     %    conv_t = 0;
     %end
     %disp({'t_dc',t_dc,'diff F-norm of M',norm(M_dc_old-M,'fro'),'conv_time',conv_t})
     
     
     M_dc_old = M;
     
     B = X*M'*W+s2;
     B_2 = ones(n,K)*(gamma-1);
     for i = 1:n 
      B(i,y(i)) = s+s2-B(i,y(i));
      B_2(i,y(i)) = gamma;
     end
     B = (B>0).*B_2;
     B = B/(n*lambda);
     if t_dc == 1 && issmm == 1
         B = zeros(n,K);
     end
     
     d = d0 + reshape(W*B'*X,(K-1)*p*q,1);
     
     %disp({'L',L});
     %disp('starting ADMM')
     %if t_dc==1
         %L = norm([A_head;d'],2);
         %disp(L)
         sigma = 100/lambda;
         omega = 100/lambda;
     %end
     % solving DC sub-problem
     sub_conv = 0;
     
     

     
     for a = 1:50
         
         z = Y(1:K*n) + sigma*reshape(Weight.*(W'*M_head*X'),K*n,1);
         z = z - sigma*(max(abs(mu+z/sigma)-1/sigma,0).*sign(mu+z/sigma)-mu);
         Y = [z;1];
         
         M_new = rho*omega*M_dc_old + M - omega*(W*reshape(v.*z,K,n)*X + reshape(d,K-1,p*q));
         for j = 1:K-1
             M_new(j,:) = reshape(svt(reshape(M_new(j,:),p,q),omega*tau)/((1+rho)*omega+1),1,p*q);
         end
         %M_new = svt(M_new/((1+rho)*omega+1),tau);
         
         theta = 1/sqrt(1+2*(1+rho)*omega);
         omega = theta*omega;
         sigma = sigma/theta;
         
         M_head = M_new + theta*(M_new-M);
         if mod(a,100)==1
             %disp(norm(M_new-M,'fro'));
             %disp({omega,sigma,theta,norm(M_new-M,'fro')/norm(M_new+M,'fro')})
         end
         if norm(M_new-M,'fro')/norm(M_new+M,'fro')< ep
             M = M_new;
             %disp({'dp break',a})
             break
         end
         M = M_new;
         
     end
     
     %disp({'t_dc',t_dc,'diff F-norm of M',norm(M_dc_old-M,'fro')})
     y_pred = RASMM_pred(X_test,M,K);
     obj_tem = RASMM_obj(M,X,y,W,p,q,n,s,s2,tau,gamma,C,K);
     accu_rate = sum(y_pred==y_test)/length(y_test);
     if mod(t_dc,1)==0
          %disp({'t_dc',t_dc,'diffnorm',norm(M_dc_old-M,'fro')/norm(M_dc_old+M,'fro'),'a',a,'accur',accu_rate,'loss_test',RASMM_loss(M,X_test,y_test,W,s,s2,gamma,K),C})
     end
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
         break
     end
     if norm(M_dc_old-M,'fro')/norm(M_dc_old+M,'fro')< 4e-4
         break;
     end
     if obj_tem >= obj_v-3e-8
         %disp({'dca break because obj increases',obj_tem - obj_v,obj_tem})
         M = M_dc_old;
         ep = ep/1000;
         %break
     else
         obj_v = obj_tem;
         accu_final = accu_rate;
         ep=ep_dc;
     end
     if ep <ep_dc*10e-9
         break;
     end
  end
end