# Source Code for Robust Multicategory Support Matrix Machines

## Solvers for the RMSMM optimization problem
We Provide two solvers in our paper, the ADMM solver in function "RMSMM_admm()" and the primal-dual solver in function "RMSMM_pd()".

## Demo
Users can run and modify the source code "simulation1_K3_admm.m" to see how the DC algorithm, ADMM algorithm and Primal-Dual algorithm work in this toolbox. 
Implementation details of the RMSMM paper, including the tunning parameters selection, running-time comparison and the validation step, are also included in this file.

The synthetic data generation procedure is done in "gen_synthetic_data_s1_K3.m"
