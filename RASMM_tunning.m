% Author: Qian Chengde 
% E-Mail: qianchd(gmail)
% Date  : 2021-10-19
% Copyright 2021 Qian Chengde.
% File: RASMM_tunning.m

% tunning for RASMM

s1 = -1;
s2 = -1/(K-1);

M_0 = zeros(K-1,p*q);
accu_SMM_best=0;
accu_final_best = 0;
tau=0;
for C=[0.001,0.005,0.01,0.05,0.1,0.5,1,5]%[1e-6,5e-6,1e-5,5e-5,1e-4,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10]
    %disp(C);
    % tau=0;
    tau=0;
    a=zeros(p,q);
    [~,~,~,~,M_SMM_tem,~,~,~,~]= RASMM_admm(X_train,y_train,X_tun,y_tun,p,q,K,2*C,M_0,gamma,s1,s2,tau,1,300,1,1);
    for k = 1:K-1
        [~,tem,~] = svd(reshape(M_SMM_tem(k,:),p,q));
        a = a +tem;
    end
    a = a/(K-1);
    
    for rank = [40:-10:30,25,15,12,6]
        tau = a(rank,rank);
        [M,~,~,~,M_SMM_tem,~,accu_SMM_tem,~,accu_final_tem]= RASMM_admm(X_train,y_train,X_tun,y_tun,p,q,K,C,M_0,gamma,s1,s2,tau,1,300,0,1);
        if accu_SMM_tem >= accu_SMM_best
            %disp({accu_SMM_tem,accu_SMM_best})
            M_SMM=M_SMM_tem;
            accu_SMM_best = accu_SMM_tem;
            tau_SMM = tau;
            C_best = C;
            a_best = a;
            disp({'tau_SMM',tau,accu_SMM_best,rank,C})
        end
        if accu_final_tem >= accu_final_best
            accu_final_best=accu_final_tem;
            M_final = M;
            C_final =C;
            disp({'tau_final',tau,'accu_final_best',accu_final_best,'accu_SMM_best',accu_SMM_best,rank,C})
        end
        M_0=M_SMM_tem;
    end
end