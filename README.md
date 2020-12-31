# SVM-Constrained Bayesian Optimization 
This repository implements the Support Vector Machine Constrained Bayesian Optimization (SVM-CBO) proposed in [1] and already used for a real-life application like Pump Scheduling Optimization in Water Distribution System in [2] and HPO task on Convolutional Neural Networks in [3-4] for Tiny Machine Learning.

# SVMCBO
Python implementation of Support Vector Machine - Constrained Bayesian Optimization (SVM-CBO)

## Packages Requirements
Execute the following command to retrieve all libraries dependecies for the software:


## Instructions to use the constrained optimization framework
To use this optimization framework you have to define the following parts:
1) In "testFunctions.R" define the objective function and insert the name of the function defined to variable "f_test" in "main_SVM-CBO.R" file
2) Insert the number of dimension of the objective function to variable "dim_f" in "main_SVM_CBO.R" file
3) Insert the values of minimum and maximum dimensions of the objective function in "real_x_min" and "real_x_max" in "main_SVM_CBO.R" file. In case the objective function has different domains values for its dimensions initialise the above variables with vectors of minimum and maximum values.
4) Decide with boolean variable "opt_min" the orientation of optimization task in "main_SVM_CBO.R" file 
5) Decide the number of optimization experiment to run with SVM-CBO with variable "nExperiments" in "main_SVM_CBO.R"
6) Execute the experiments running the "main_SVM_CBO.R" file

At the end of "main_SVM_CBO.R" execution all the experiments results are save in "SVM-CBO_results_experiments.RData" file.

## References

[1] Candelieri, A. (2019). Sequential model based optimization of partially defined functions under unknown constraints. Journal of Global Optimization, 1-23. (https://link.springer.com/article/10.1007/s10898-019-00860-4)

[2] Candelieri, A., Galuzzi, B., Giordani, I., Perego, R., & Archetti, F. (2019, May). Optimizing partially defined black-box functions under unknown constraints via Sequential Model Based Optimization: an application to Pump Scheduling Optimization in Water Distribution Networks. In International Conference on Learning and Intelligent Optimization (pp. 77-93). Springer, Cham. (https://link.springer.com/chapter/10.1007/978-3-030-38629-0_7)

[3] Perego, R., Candelieri, A., Archetti, F., & Pau, D. (2020, September). Tuning Deep Neural Network’s Hyperparameters Constrained to Deployability on Tiny Systems. In International Conference on Artificial Neural Networks (pp. 92-103). Springer, Cham.

[4] Perego, R., Candelieri A., Archetti F., \& Pau D. AutoTinyML for microcontrollers: dealing with black-box deployability. IEEE Transactions on Neural Networks and Learning Systems (2020). (Under review)
