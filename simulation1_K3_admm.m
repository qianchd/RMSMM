% Author: Qian Chengde 
% E-Mail: qianchd(gmail)
% Date  : 2021-10-19
% Copyright 2021 Qian Chengde.
% File: simulation1_K3_admm.m

% Simulation 1 of RAMSMM with ADMM method and K = 3

% replication times
t = 1;

res_rasmm_ts_final_10 = zeros(t,1);
res_SMM_10 = zeros(t,1);
res_1v1_10 = zeros(t,1);
res_C_10 = zeros(t,1);
res_final_ramsvm_t0_10 = zeros(t,1);
res_SMM_ramsvm_t0_10 = zeros(t,1);
res_final_ramsvm_ts_10 = zeros(t,1);
res_SMM_ramsvm_ts_10 = zeros(t,1);
time_svm_10 = zeros(t,1);
time_SMM_10 = zeros(t,1);

res_rasmm_ts_final_20 = zeros(t,1);
res_SMM_20 = zeros(t,1);
res_1v1_20 = zeros(t,1);
res_C_20 = zeros(t,1);
res_final_ramsvm_t0_20 = zeros(t,1);
res_SMM_ramsvm_t0_20 = zeros(t,1);
res_final_ramsvm_ts_20 = zeros(t,1);
res_SMM_ramsvm_ts_20 = zeros(t,1);
time_svm_20 = zeros(t,1);
time_SMM_20 = zeros(t,1);

res_rasmm_ts_final_0 = zeros(t,1);
res_SMM_0 = zeros(t,1);
res_1v1_0 = zeros(t,1);
res_C_0 = zeros(t,1);
res_final_ramsvm_t0_0 = zeros(t,1);
res_SMM_ramsvm_t0_0 = zeros(t,1);
res_final_ramsvm_ts_0 = zeros(t,1);
res_SMM_ramsvm_ts_0 = zeros(t,1);
time_svm_0 = zeros(t,1);
time_SMM_0 = zeros(t,1);

for sigma = [0.5,0.7,0.9]
    
    for m = 1:t
        disp({'m',m});
        
        gen_synthetic_data_s1_K3;
        
        sn=1000;
        
        %r=ceil(rand(rn,1)*K);
        y_train = y(1:sn);
        %y_train(1:rn)=r;
        X_train = X(1:sn,:);
        %for i = 1:rn
        %    X_tem(i,:)=3*G(1,:) + normrnd(0,sigma,1,p*q);
        %end
        X_tun = X(1001:11000,:);
        y_tun = y(1001:11000);
        X_test = X(11001:end,:);
        y_test = y(11001:end);
        
        %#################### with 0 percent outliers ################
        % RAMSMM
        gamma = 0.5;
        tic;
        RASMM_tunning;
        time_SMM_0(m) = toc;
        accu_SMM_best = mean(y_test==RASMM_pred(X_test,M_SMM,K));
        accu_final_best = mean(y_test==RASMM_pred(X_test,M_final,K));
        disp({'accu_final & SMM',accu_final_best,accu_SMM_best,C_final,C_best,time_SMM_0(m)})
        res_rasmm_ts_final_0(m) = accu_final_best;
        res_SMM_0(m) = accu_SMM_best;
        
        % RAMSVM_ts
        tic;
        M_0 = zeros(K-1,p*q);
        tau = 0;
        accu_SMM=0;
        accu_final = 0;
        for C=[1e-6,5e-6,1e-5,5e-5,1e-4,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,25];
            [M,S_head,A,W,M_SMM_tem,M_2,accu_SMM_tem,accu_2_tem,accu_final_tem]= RASMM_admm(X_train,y_train,X_tun,y_tun,p,q,K,2*C,M_0,gamma,s1,s2,tau,1,300,0,1);
            if accu_SMM_tem >= accu_SMM
                accu_SMM = accu_SMM_tem;
                M_SMM = M_SMM_tem;
            end
            if accu_final_tem >= accu_final
                accu_final = accu_final_tem;
                M_final = M;
            end
        end
        accu_SMM = mean(y_test==RASMM_pred(X_test,M_SMM,K));
        accu_final = mean(y_test==RASMM_pred(X_test,M_final,K));
        time_svm_0(m)=toc; %%%time time
        disp({'rmsvm svm&tsvm',accu_SMM,accu_final,time_svm_0(m)})
        res_SMM_ramsvm_ts_0(m) = accu_SMM;
        res_final_ramsvm_ts_0(m) = accu_final;
        
        %#################### with 10 percent outliers ################
        rn=sn/10;
        disp({'m',m});
        for i = 1:rn
            X_train(i,:)=3*G(1,:) + normrnd(0,sigma,1,p*q);
        end
        
        % RAMSMM
        gamma = 0.5;
        tic;
        RASMM_tunning;
        time_SMM_10(m) = toc;
        accu_SMM_best = mean(y_test==RASMM_pred(X_test,M_SMM,K));
        accu_final_best = mean(y_test==RASMM_pred(X_test,M_final,K));
        disp({'accu_final & SMM',accu_final_best,accu_SMM_best,C_final,C_best,time_SMM_10(m)})
        res_rasmm_ts_final_10(m) = accu_final_best;
        res_SMM_10(m) = accu_SMM_best;
        %disp({m,'gamma',gamma})
        %gamma = 1;
        %synthesis_data2;
        %res_rasmm_ts_final_odinary_0(m) = accu_final;
        %res_SMM_odinary_0(m) = accu_SMM;
        %disp({'gamma',gamma})
        
        
        % RAMSVM
        tic;
        M_0 = zeros(K-1,p*q);
        tau = 0;
        accu_SMM=0;
        accu_final = 0;
        for C=[1e-6,5e-6,1e-5,5e-5,1e-4,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,25];
            
            [M,S_head,A,W,M_SMM_tem,M_2,accu_SMM_tem,accu_2_tem,accu_final_tem]= RASMM_admm(X_train,y_train,X_tun,y_tun,p,q,K,2*C,M_0,gamma,s1,s2,tau,1,300,0,1);
            if accu_SMM_tem >= accu_SMM
                accu_SMM = accu_SMM_tem;
                M_SMM = M_SMM_tem;
            end
            if accu_final_tem >= accu_final
                accu_final = accu_final_tem;
                M_final = M;
            end
        end
        accu_SMM = mean(y_test==RASMM_pred(X_test,M_SMM,K));
        accu_final = mean(y_test==RASMM_pred(X_test,M_final,K));
        time_svm_10(m)=toc; %%%time time
        disp({'rmsvm svm&tsvm',accu_SMM,accu_final,time_svm_10(m)})
        res_SMM_ramsvm_ts_10(m) = accu_SMM;
        res_final_ramsvm_ts_10(m) = accu_final;
        
        
        %#################### with 20 percent outliers ################
        rn=sn/5;
        disp({'m',m});
        for i = 1:rn
            %y_train(i) = ceil(rand(n,1)*K);
            X_train(i,:)=3*G(1,:) + normrnd(0,sigma,1,p*q);
        end
        
        % RAMSMM
        gamma = 0.5;
        tic;
        RASMM_tunning;
        time_SMM_20(m) = toc;
        accu_SMM_best = mean(y_test==RASMM_pred(X_test,M_SMM,K));
        accu_final_best = mean(y_test==RASMM_pred(X_test,M_final,K));
        disp({'accu_final & SMM',accu_final_best,accu_SMM_best,C_final,C_best,time_SMM_20(m)})
        res_rasmm_ts_final_20(m) = accu_final_best;
        res_SMM_20(m) = accu_SMM_best;
        %disp({m,'gamma',gamma})
        %gamma = 1;
        %synthesis_data2;
        %res_rasmm_ts_final_odinary_0(m) = accu_final;
        %res_SMM_odinary_0(m) = accu_SMM;
        %disp({'gamma',gamma})
        
        
        % RAMSVM
        tic;
        M_0 = zeros(K-1,p*q);
        tau = 0;
        accu_SMM=0;
        accu_final = 0;
        for C=[1e-6,5e-6,1e-5,5e-5,1e-4,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,25];
            
            [M,S_head,A,W,M_SMM_tem,M_2,accu_SMM_tem,accu_2_tem,accu_final_tem]= RASMM_admm(X_train,y_train,X_tun,y_tun,p,q,K,2*C,M_0,gamma,s1,s2,tau,1,300,0,1);
            if accu_SMM_tem >= accu_SMM
                accu_SMM = accu_SMM_tem;
                M_SMM = M_SMM_tem;
            end
            if accu_final_tem >= accu_final
                accu_final = accu_final_tem;
                M_final = M;
            end
        end
        accu_SMM = mean(y_test==RASMM_pred(X_test,M_SMM,K));
        accu_final = mean(y_test==RASMM_pred(X_test,M_final,K));
        time_svm_20(m)=toc; %%%time time
        disp({'rmsvm svm&tsvm',accu_SMM,accu_final,time_svm_20(m)})
        res_SMM_ramsvm_ts_20(m) = accu_SMM;
        res_final_ramsvm_ts_20(m) = accu_final;
        columns = {'tramsmm_ts', 'ramsmm', 'tramsvm_ts', 'ramsvm','time_TSMM','time_tsvm'};
        data_20 = table(res_rasmm_ts_final_20, res_SMM_20, res_final_ramsvm_ts_20,  res_SMM_ramsvm_ts_20,time_SMM_20,time_svm_20, 'VariableNames', columns);
        data_10 = table(res_rasmm_ts_final_10, res_SMM_10, res_final_ramsvm_ts_10, res_SMM_ramsvm_ts_10,time_SMM_10,time_svm_10, 'VariableNames', columns);
        data_0 = table(res_rasmm_ts_final_0, res_SMM_0, res_final_ramsvm_ts_0, res_SMM_ramsvm_ts_0,time_SMM_0,time_svm_0, 'VariableNames', columns);
        writetable(data_20, ['res_',num2str(sigma),'_20.csv'])
        writetable(data_10, ['res_',num2str(sigma),'_10.csv'])
        writetable(data_0, ['res_',num2str(sigma),'_0.csv'])
    end
end