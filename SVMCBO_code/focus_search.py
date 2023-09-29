import numpy as np
from pyDOE2 import lhs
from joblib import Parallel, delayed


def focus_search(f, args, n_restart = 3, n_focus = 5):
    bounds = np.array([[0, 1], [0, 1]])
    cand_points = []
    cand_acq = []

    for idx_start in range(0, n_restart):
        optimal_point = optimal_value = []
        new_bounds = bounds

        for iter_n in range(0,n_focus):
            x = lhs(len(new_bounds), 5000)
            ## don't use har-coded parallelism, TODO!
#            y = Parallel(n_jobs=16)(delayed(f)([v], args) for v in x)  # evaluate points in parallel
            y = f(x, args)
            x_star = x[np.argmin(y)]
            y_star = np.min(y)
            optimal_point = optimal_point + [x_star]
            optimal_value = optimal_value + [y_star]
            new_bounds = []

            for i in range(len(bounds)):
                l_xi = np.min(x[:, i])
                u_xi = np.max(x[:, i])
                new_l_xi = np.max([l_xi, x_star[i] - 0.25 * (u_xi - l_xi)])
                new_u_xi = np.min([u_xi, x_star[i] + 0.25 * (u_xi - l_xi)])
                new_bounds = new_bounds + [[new_l_xi, new_u_xi]]

        optimal_value = np.array(optimal_value)
        optimal_point = optimal_point[np.argmin(optimal_value)]
        optimal_value = np.min(optimal_value)

        cand_points = cand_points + [optimal_point]
        cand_acq = cand_acq + [optimal_value]

    final_cand_point = cand_points[np.argmin(cand_acq)]
    final_cand_acq = np.min(cand_acq)
    return (final_cand_point, final_cand_acq)

def focus_search_parallel(f, args, n_restart = 3, n_focus = 5):
    bounds = args["bounds"]
    ##TODO: avoid hard-coded parallelism
    results = Parallel(n_jobs=6)(delayed(focusing)(f, bounds, n_focus, args) for i in np.arange(0,n_restart))
    cand_xs = np.array([r[0] for r in results])
    cand_acqs = np.array([r[1] for r in results])
    classifier = args["classifier"]
    labels_cand = classifier.predict(cand_xs)
    next_x = cand_xs[np.argmin(cand_acqs)]
    #print("Next real point focus {}".format(next_x))
    if len(np.where(labels_cand==0)[0]) == 1: ## nel caso focus torni un valore classificato come feasible
        next_x = cand_xs[np.where(labels_cand == 0)][0]
        #print("Check 1 next_x {}".format(next_x))
    elif len(np.where(labels_cand==0)[0]) > 1: ## nel caso focus torni più di un valore classificato come feasible
        cand_xs_pos = cand_xs[np.where(labels_cand==0)[0]]
        cand_acqs_pos = cand_acqs[np.where(labels_cand==0)[0]]
        next_x = cand_xs_pos[np.argmin(cand_acqs_pos)]
        #print("Check 2 next_x {}".format(next_x))
    else: ## nel caso in cui siano stati proposti solo punti non feasible secondo classificatore
        x = lhs(len(bounds), 10000)
        labels_cand = classifier.predict(x)
        #print("Check 3 next_x {}".format(next_x))
        if len(np.where(labels_cand == 1)[0]) == 10000:
            print(len(np.where(labels_cand == 1)[0]))
            next_x = x[-1]
        else:
            x = x[np.where(labels_cand == 0)]
            values = f(x,args)
            next_x = x[np.argmin(values)]
        #print("Check 3.1 Real next_x {}".format(next_x))
    return next_x

def focusing(f, b, n_iter, args):
    optimal_point = optimal_value = []
    new_bounds = b
    for iter_n in range(0, n_iter):
        x = lhs(len(new_bounds), 10000)
        classifier = args["classifier"]
        labels_cand = classifier.predict(x)
        if len(np.where(labels_cand == 1)[0]) == 10000:
            print(len(np.where(labels_cand == 1)[0]))
            return x[-1], np.inf
        
        x = x[np.where(labels_cand == 0)]
        y = f(x, args)# call acquisition function
        x_star = x[np.argmin(y)]
        y_star = np.min(y)
        optimal_point = optimal_point + [x_star]
        optimal_value = optimal_value + [y_star]
        new_bounds = []
        for i in range(len(b)):
            l_xi = np.min(x[:, i])
            u_xi = np.max(x[:, i])
            new_l_xi = np.max([l_xi, x_star[i] - 0.25 * (u_xi - l_xi)])
            new_u_xi = np.min([u_xi, x_star[i] + 0.25 * (u_xi - l_xi)])
            new_bounds = new_bounds + [[new_l_xi, new_u_xi]]
        #print("Shrinked Bounds: {}".format(new_bounds))
    optimal_value = y_star
    optimal_point = x_star
    return optimal_point, optimal_value

