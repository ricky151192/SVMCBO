
import matplotlib.pyplot as plt
from SVMCBO_code.SVMCBO import SVMCBO
from test_function_suite import *

## Example optimization Mishra Bird test function ############################################
test_f = mishra_bird_c
optimum_for_gap = -106.7645367

## Experiment with SVM-CBO
exp_svmcbo = SVMCBO(f=test_f, surrogate="GP")
exp_svmcbo.init_opt()
exp_svmcbo.phase1()
exp_svmcbo.phase2()
res = exp_svmcbo.generate_result()
print(f"Optimum point: {res.get('x')}")
print(f"Optimal value: {res.get('fun')}")
gap_metric_svmcbo = exp_svmcbo.gap_metric(optimum_value=optimum_for_gap)

## Experiment with SVM-CBO_RF
exp_svmcbo_rf = SVMCBO(f=test_f, surrogate="RF")
exp_svmcbo_rf.init_opt()
exp_svmcbo_rf.phase1()
exp_svmcbo_rf.phase2()
res = exp_svmcbo_rf.generate_result()
print(f"Optimum point: {res.get('x')}")
print(f"Optimal value: {res.get('fun')}")
gap_metric_svmcbo_rf = exp_svmcbo_rf.gap_metric(optimum_value=optimum_for_gap)

## Comparison gap metric between SVMCBO and SVMCBO_RF

plt.plot(gap_metric_svmcbo, label="SVM-CBO")
plt.plot(gap_metric_svmcbo_rf, label="SVM-CBO_RF")
plt.ylim(0.0,1.1)
plt.ylabel("Gap Metric")
plt.xlabel("Iterations")
plt.title("Comparison gap metric on {} test function".format(test_f.__name__))
plt.legend(loc="best")
plt.show()

##################################################################################################

## Example optimization Michalewicz test function ############################################
test_f = michalewicz_c
optimum_for_gap = -1.801140718473825

## Experiment with SVM-CBO
exp_svmcbo = SVMCBO(f=test_f, surrogate="GP")
exp_svmcbo.init_opt()
exp_svmcbo.phase1()
exp_svmcbo.phase2()
res = exp_svmcbo.generate_result()
print(f"Optimum point: {res.get('x')}")
print(f"Optimal value: {res.get('fun')}")
gap_metric_svmcbo = exp_svmcbo.gap_metric(optimum_value=optimum_for_gap)

## Experiment with SVM-CBO_RF
exp_svmcbo_rf = SVMCBO(f=test_f, surrogate="RF")
exp_svmcbo_rf.init_opt()
exp_svmcbo_rf.phase1()
exp_svmcbo_rf.phase2()
res = exp_svmcbo_rf.generate_result()
print(f"Optimum point: {res.get('x')}")
print(f"Optimal value: {res.get('fun')}")
gap_metric_svmcbo_rf = exp_svmcbo_rf.gap_metric(optimum_value=optimum_for_gap)

## Comparison gap metric between SVMCBO and SVMCBO_RF

plt.plot(gap_metric_svmcbo, label="SVM-CBO")
plt.plot(gap_metric_svmcbo_rf, label="SVM-CBO_RF")
plt.ylim(0.0,1.1)
plt.ylabel("Gap Metric")
plt.xlabel("Iterations")
plt.title("Comparison gap metric on {} test function".format(test_f.__name__))
plt.legend(loc="best")
plt.show()

##################################################################################################