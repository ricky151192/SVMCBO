
from sklearn.svm import SVC
from .my_gp import *
from .my_rf import *
from sklearn.gaussian_process.kernels import RBF
from .svmcbo_utils import *
from .focus_search import *
from pyDOE2 import lhs
import numpy as np

class SVMCBO():
    def __init__(self, f, bounds=[[0, 1], [0, 1]], n_init=10,
                 iter_phase1=60, iter_phase2=30,
                 surrogate="GP",
                 noise=None, gamma=0.1, classifier=SVC, seed=42):
        self.noise = noise
        self.classifier = classifier
        self.gamma = gamma
        self.gamma_svm = "scale"
        self.x_tot = []
        self.y_tot = []
        self.x_feasible = []
        self.y_feasible = []
        self.labels = []
        self.n_init = n_init
        self.iter_phase1 = iter_phase1
        self.iter_phase2 = iter_phase2
        self.surrogate_type = surrogate
        self.surrogates = list()
        self.bounds = bounds
        self.f = f
        self.classifiers = list()
        self.score_svm = list()
        self.seed = seed

    def init_opt(self):
        x = lhs(len(self.bounds), self.n_init, random_state=self.seed)
        self.labels = np.zeros((x.shape[0],))
        self.y_feasible = list()
        i = 0
        while i < x.shape[0]:
            print("*** Iterazione #", i)
            print("==> Points to evaluate: ", x[i,])
            function_evaluation = self.f(x[i,])

            if np.isnan(function_evaluation):
                print("---------- NO FUNCTION EVALUATION ----------")
                self.labels[i] = 1
            else:
                print("==> Function evaluation:")
                print(function_evaluation)
                self.labels[i] = 0
                self.y_feasible = self.y_feasible + [function_evaluation]

            self.y_tot = self.y_tot + [function_evaluation]
            i = i + 1
            print(" --> Numero di punti feasible:", len(self.y_feasible))

        self.x_tot = x.tolist()

    def phase1(self):
        print(self.gamma)
        self.gamma = 1/(2*np.var(self.x_tot))
        svm = estimateFeasibleRegion(self.x_tot, self.labels, self.gamma)
        score_svm = svm.score(self.x_tot, self.labels)
        self.score_svm.append(score_svm)
        print('*'*70)
        print("> Score SVM: {}".format(score_svm))
        print('*'*70)
        self.classifiers.append(svm)
        print(svm.predict(self.x_tot))
        print(self.labels)
        self.classifier = svm

        print("Start phase 1!!!")
        for i in np.arange(0, self.iter_phase1):
            print('*' * 70)
            print("Iterazione", i, "in fase 1... (punti feasible:", len(self.y_feasible), ")")
            print("[Current best value: ", np.min(self.y_feasible),"]")
            print("Computing the new point...")
            x_new = nextPointPhase1(sampledPoints=self.x_tot, svm=self.classifiers[i], gamma=np.var(self.x_tot),
                                    dimensions_test=self.bounds)
            print("Updating the design...")
            self.x_tot = self.x_tot + x_new.tolist()
            # labels = evaluateEstimatedFeasibility(x, svm)
            print("==> Points to evaluate: ", x_new)
            function_evaluation = self.f(x_new[0])
            print("-------------------")
            print(function_evaluation)
            print("-------------------")
            if np.isnan(function_evaluation):
                print("==> Function evaluation: ", function_evaluation)
                new_label = 1
            else:
                new_label = 0
                self.y_feasible = self.y_feasible + [function_evaluation]

            self.y_tot = self.y_tot + [function_evaluation]
            self.labels = np.concatenate((self.labels, np.array([new_label])))

            print("Updating the estimated feasible region...")
            self.gamma = 1 / (2 * np.var(self.x_tot))
            svm = estimateFeasibleRegion(self.x_tot, self.labels, self.gamma)
            score_svm = svm.score(self.x_tot, self.labels)
            print("> Score SVM: {}".format(score_svm))
            print('*' * 70)
            self.score_svm.append(score_svm)
            self.classifiers.append(svm)
        self.x_feasible = np.array(self.x_tot)[np.where(self.labels == 0)[0],].tolist()

    def phase2(self):
        warnings.filterwarnings("ignore")
        for iter_bo in np.arange(0, self.iter_phase2):
            print('*' * 70)
            print("> Iterazione {} in fase 2... (punti feasible:{})".format(iter_bo, len(self.y_feasible)))
            print("[Current best value: ", np.min(self.y_feasible),"]")
            x_ = self.x_feasible
            y_ = self.y_feasible

            svm_model = self.classifiers[-1]
            if self.surrogate_type == "GP":
                surrogate_model = GaussianProcessRegressor(RBF())
            elif self.surrogate_type == "RF":
                surrogate_model = RandomForestRegressor()
            else:
                surrogate_model = ExtraTreesRegressor()

            surrogate_model.fit(x_, y_)
            self.surrogates.append(surrogate_model)
            params = {'model': surrogate_model, 'classifier': svm_model, 'bounds': self.bounds, 'n_sampling': 10000}
            next_x = focus_search_parallel(f=acquisition_function, args=params)
            value = self.f(next_x)
            print("f({}) = {}".format(next_x, value))
            if np.isnan(value):
                new_label = 1
            else:
                new_label = 0
            ### Add classification label new point
            # x = np.concatenate((x, np.array([next_x])))
            self.x_tot = self.x_tot + [next_x.tolist()]
            self.labels = np.concatenate((self.labels, np.array([new_label])))

            ### Update surrogate model in case unfeasible point sampled!
            if np.isnan(value):
                print("**** Retraining the boundaries! ***")
                self.gamma = 1 / (2 * np.var(self.x_tot))
                svm = estimateFeasibleRegion(self.x_tot, self.labels,self.gamma)
                score_svm = svm.score(self.x_tot, self.labels)
                print("> Score SVM: {}".format(score_svm))
                self.score_svm.append(score_svm)
                self.classifiers.append(svm)
            else:
                self.classifiers.append(self.classifiers[-1])  # continuo ad appendere svm migliore!
                score_svm = self.classifiers[-1].score(self.x_tot, self.labels)
                print("> Score SVM: {}".format(score_svm))
                self.score_svm.append(score_svm)
                # y = np.concatenate((y, np.array([value])))
                self.y_feasible = self.y_feasible + [value]
                self.x_feasible = self.x_feasible + [next_x.tolist()]
            print('*' * 70)
            self.y_tot = self.y_tot + [value]

    ## Generate results of experiment
    def generate_result(self):
        x = self.x_feasible[np.argmin(self.y_feasible)]
        fun = np.min(self.y_feasible)
        return {"x": x, "fun": fun, "xs": self.x_tot, "funs": self.y_tot, "xs_feasible": self.x_feasible,
                "funs_feasible": self.y_feasible}

    ## Generate gap metric for check the progress of the optimizer into optimization process
    def gap_metric(self, optimum_value=0):
        y_tmp = [9999999 if np.isnan(x) else x for x in self.y_tot]
        current_opt = np.min(y_tmp[0:self.n_init])
        init_opt = np.min(y_tmp[0:self.n_init])
        gap_metric = []
        for i in range(0, len(y_tmp)):
            if current_opt > y_tmp[i]:
                current_opt = y_tmp[i]
            gap = abs(current_opt - init_opt) / abs(optimum_value - init_opt)
            gap_metric = gap_metric + [gap]
        return gap_metric