# Created by ricky at 15/12/2019

import numpy as np
import scipy as spy
from joblib import Parallel, delayed
from pyDOE2 import lhs
import warnings

def estimateFeasibleRegion(x,y, gamma):
    from sklearn.svm import SVC
    svm_model = SVC(gamma=gamma)
    svm_model.fit(x, y)
    return svm_model

def evaluateEstimatedFeasibility(x, svm):
    y = svm.predict(x)
    return y

def coverage(x, sampledPoints, gamma):
    x = np.array(x)
    sampledPoints= np.array(sampledPoints)
    C = 0
    for i in np.arange(0, len(sampledPoints)):
        C = C + np.exp(-np.sum((x - sampledPoints[i, ]) ** 2) / (2 * gamma**2))
    return C

def boundness(x, svm):
    B = abs(svmSurface(x, svm))
    return B

def svmSurface(x,svm):
    x = np.array(x)
    coeffs = svm.dual_coef_[0]
    svs = svm.support_vectors_
    gamma = svm.gamma # because selected gamma='auto' in sklearn SVC
    sigma = 1/gamma
    f = 0
    for k in np.arange(0,len(coeffs)):
        f = f + coeffs[k] * np.exp(-sigma * (np.sum((x-svs[k, ])**2)))
    return f

def phase1AcquisitionFunction(x, args):

    sampledPoints = args["sampledPoints"]
    svm = args["svm"]
    gamma = args["gamma"]
    C = coverage(x, sampledPoints, gamma)
    B = boundness(x, svm)
    value = C+B
    return value

def nextPointPhase1(sampledPoints, svm, gamma, dimensions_test):

    nStartingPoints = 30
    #xlimits = np.array(dimensions_test)
    startingPoints = lhs(len(dimensions_test), nStartingPoints)
    additional = {'sampledPoints': sampledPoints, 'svm': svm, 'gamma': gamma} # optional parameters for minimization scipy
    locOptX = list()
    locOptY = list()

    # for i in np.arange(0,nStartingPoints):
    #     #print("----", i, "----")
    #     #print("Starting point:",startingPoints[i,])
    #     ## Dovrebbero tornare un valore di funzione e il sample x migliore da valutare..
    #     opt = spy.optimize.minimize(fun=phase1AcquisitionFunction,
    #                                 args=additional,
    #                                 x0=startingPoints[i,],
    #                                 bounds=dimensions_test,#((-1.5, 1.5),(-0.5,2.5)),
    #                                 method='L-BFGS-B'#method='BFGS'#method='Nelder-Mead'
    #                                 )
    #     locOptX = locOptX + [opt.x]
    #     locOptY = locOptY + [phase1AcquisitionFunction(opt.x, additional)]
    #     #print(locOptY[-1])
    # ix = np.where(np.array(locOptY) == np.min(np.array(locOptY)))[0][0]
    #
    # print(ix)

    ##TODO: avoid hard-coded parallelism
    results = Parallel(6)(
        delayed(spy.optimize.minimize)(
            fun=phase1AcquisitionFunction,
            args=additional,
            x0=x0,
            bounds=dimensions_test,
            method='L-BFGS-B',
            options={'maxiter':100})
        for x0 in startingPoints)

    for ix in range(len(results)):
        locOptX = locOptX + [results[ix]['x']]
        locOptY = locOptY + [results[ix]['fun']]

    ix = np.where(np.array(locOptY) == np.min(np.array(locOptY)))[0][0]

    return (np.array([locOptX[ix]]))


def acquisition_function(x, args, beta=1.96):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model = args["model"]
        classifier = args["classifier"]
        mu, std = model.predict(x, return_std=True)
        labels = classifier.predict(x)
        label_neg = np.where(labels == 1)[0]
        label_pos = np.where(labels == 0)[0]
        #print("* Punti PREDETTI 'feasible:' {} su un totale di {}".format(len(label_pos), len(x)))
        mu[label_neg] = np.max(mu)
        lcb = mu - beta * std
        return lcb

# def acquisition_function_penalty(x, args, beta=1.96):
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#
#         model = args["model"]
#         mu, std = model.predict(x, return_std=True)
#         lcb = mu - beta * std
#         return lcb
