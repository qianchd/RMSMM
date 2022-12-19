# Source Code for Robust Multicategory Support Matrix Machines

Here is the algorithm implementation of the paper "Robust multicategory support matrix machines." Mathematical programming 176.1 (2019): 429-463.

## Solvers for the RMSMM optimization problem
We provide two solvers in our paper, the ADMM solver in function "RMSMM_admm()" and the primal-dual solver in function "RMSMM_pd()". And deal with the Truncated loss by the DC algorithm.

## Demo
One can run and modify the source code "simulation1_K3_admm.m" to see how the DC algorithm, ADMM algorithm and Primal-Dual algorithm work in this toolbox. 
Implementation details of the RMSMM paper, including the tunning parameters selection, running-time comparison and the validation step, are also included in this file.

The synthetic data generation procedure is done in "gen_synthetic_data_s1_K3.m"
