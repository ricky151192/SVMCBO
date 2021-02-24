# Support Vector Machine - Constrained Bayesian Optimization 
This repository implements the Python version of Support Vector Machine - Constrained Bayesian Optimization (SVM-CBO) proposed in [1] and already used for a real-life applications like Pump Scheduling Optimization in Water Distribution System in [2] and HPO task on Convolutional Neural Networks in [3-4]\* for Tiny Machine Learning.

## Packages Requirements
The Python version currently supported is 3.7.  
The requirements to use this library are contained inside of "requirements_SVMCBO.txt".  
Can be used pip command to install all requirements.

## Instructions to use the constrained optimization framework
A script named "ExampleOnTestFunction.py reports an example of how to use SVMCBO framework on two simple test functions.

## References

[1] Candelieri, A. (2019). Sequential model based optimization of partially defined functions under unknown constraints. Journal of Global Optimization, 1-23. (https://link.springer.com/article/10.1007/s10898-019-00860-4)

[2] Candelieri, A., Galuzzi, B., Giordani, I., Perego, R., & Archetti, F. (2019, May). Optimizing partially defined black-box functions under unknown constraints via Sequential Model Based Optimization: an application to Pump Scheduling Optimization in Water Distribution Networks. In International Conference on Learning and Intelligent Optimization (pp. 77-93). Springer, Cham. (https://link.springer.com/chapter/10.1007/978-3-030-38629-0_7)

[3] Perego, R., Candelieri, A., Archetti, F., & Pau, D. (2020, September). Tuning Deep Neural Network’s Hyperparameters Constrained to Deployability on Tiny Systems. In International Conference on Artificial Neural Networks (pp. 92-103). Springer, Cham. (https://link.springer.com/chapter/10.1007/978-3-030-61616-8_8)

[4] Perego, R., Candelieri A., Archetti F., \& Pau D. AutoTinyML for microcontrollers: dealing with black-box deployability. Expert Systems With Applications (2021). (Under review)

\* A modified version of SVM-CBO is implemented to support HPO task on Keras Convolutional Neural Networks.
