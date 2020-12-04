"""
Mostly copied from wandb client code
Modified "next_sample" code to do the following:
-accepts a 'failure_cost' argument
-if failure cost 'c' is nonzero, modifies expected improvement of each
 sample according to:
   e' = p e / (p (1-c) + c)
 where 'p' is probability of success and 'e' is unmodified expected improvement
-returns expected improvements for whole sample

Bayesian Search
Check out https://arxiv.org/pdf/1206.2944.pdf
 for explanation of bayesian optimization
We do bayesian optimization and handle the cases where some X values are integers
as well as the case where X is very large.
"""

import numpy as np
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import Matern
#import scipy.stats as stats
import math
from wandb.util import get_module
from wandb.sweeps.base import Search
from wandb.sweeps.params import HyperParameter, HyperParameterSet

sklearn_gaussian = get_module('sklearn.gaussian_process')
sklearn_linear = get_module('sklearn.linear_model')
sklearn_svm = get_module('sklearn.svm')
sklearn_discriminant = get_module('sklearn.discriminant_analysis')
scipy_stats = get_module('scipy.stats')


def fit_normalized_gaussian_process(X, y, nu=1.5):
    """
        We fit a gaussian process but first subtract the mean and divide by stddev.
        To undo at prediction tim, call y_pred = gp.predict(X) * y_stddev + y_mean
    """
    gp = sklearn_gaussian.GaussianProcessRegressor(
        kernel=sklearn_gaussian.kernels.Matern(nu=nu), n_restarts_optimizer=2, alpha=0.0000001, random_state=2
    )
    if len(y) == 1:
        y = np.array(y)
        y_mean = y[0]
        y_stddev = 1
    else:
        y_mean = np.mean(y)
        y_stddev = np.std(y) + 0.0001
    y_norm = (y - y_mean) / y_stddev
    gp.fit(X, y_norm)
    return gp, y_mean, y_stddev
    
def train_logistic_regression(X, y):
    lr = sklearn_linear.LogisticRegression()
    lr.fit(X, y.astype(int))
    return lambda X : lr.predict_proba(X)[...,1], 0, 1
    
def train_rbf_svm(X, y):
    svc = sklearn_svm.SVC(probability=True)
    svc.fit(X, y.astype(int))
    return lambda X : svc.predict_proba(X)[...,1], 0, 1
    
def train_qda(X,y):
    qda = sklearn_discriminant.QuadraticDiscriminantAnalysis()
    qda.fit(X, y.astype(int))
    return lambda X : qda.predict_proba(X)[...,1], 0, 1
    


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def random_sample(X_bounds, num_test_samples):
    num_hyperparameters = len(X_bounds)
    test_X = np.empty((num_test_samples, num_hyperparameters))
    for ii in range(num_test_samples):
        for jj in range(num_hyperparameters):
            if type(X_bounds[jj][0]) == int:
                assert (type(X_bounds[jj][1]) == int)
                test_X[ii, jj] = np.random.randint(
                    X_bounds[jj][0], X_bounds[jj][1])
            else:
                test_X[ii, jj] = np.random.uniform() * (
                    X_bounds[jj][1] - X_bounds[jj][0]
                ) + X_bounds[
                    jj
                ][
                    0
                ]
    return test_X


def predict(X, y, test_X, nu=1.5):
    gp, norm_mean, norm_stddev = fit_normalized_gaussian_process(X, y, nu=nu)
    y_pred, y_std = gp.predict([test_X], return_std=True)
    y_std_norm = y_std * norm_stddev
    y_pred_norm = (y_pred * norm_stddev) + norm_mean
    return y_pred_norm[0], y_std_norm[0]


def train_runtime_model(sample_X, runtimes, X_bounds, nu=1.5, model='gaussian'):
    if sample_X.shape[0] != runtimes.shape[0]:
        raise ValueError("Sample X and runtimes must be the same length")

    if model=='gaussian':
        return train_gaussian_process(sample_X, runtimes, X_bounds, nu=nu)
    elif model=='logistic' and runtimes.any() and not runtimes.all():
        return train_logistic_regression(sample_X, runtimes)
    elif model=='rbf_svm' and runtimes.any() and not runtimes.all():
        return train_rbf_svm(sample_X, runtimes)
    elif model=='qda' and runtimes.sum() > 1 and runtimes.sum() < len(runtimes) - 1:
        return train_qda(sample_X, runtimes)
    else:
        return None, 0, 1


#def train_failure_model(sample_X, failures, X_bounds):
#    if sample_X.shape[0] != failures.shape[0]:
#        raise ValueError("Sample X and runtimes must be the same length")
#
#    return train_gaussian_process(sample_X, runtimes, X_bounds)


def train_gaussian_process(
    sample_X, sample_y, X_bounds, current_X=None, nu=1.5, max_samples=100
):
    """
    Trains a Gaussian Process function from sample_X, sample_y data
    Handles the case where there are other training runs in flight (current_X)
        Arguments:
            sample_X - vector of already evaluated sets of hyperparameters
            sample_y - vector of already evaluated loss function values
            X_bounds - minimum and maximum values for every dimension of X
            current_X - hyperparameters currently being explored
            nu - input to the Matern function, higher numbers make it smoother 0.5, 1.5, 2.5 are good values
             see http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html
        Returns:
            gp - the gaussian process function
            y_mean - mean
            y_stddev - stddev
            To make a prediction with gp on real world data X, need to call:
            (gp.predict(X) * y_stddev) + y_mean
    """
    if current_X is not None:
        current_X = np.array(current_X)
        if len(current_X.shape) != 2:
            raise ValueError("Current X must be a 2 dimensional array")

        # we can't let the current samples be bigger than max samples
        # because we need to use some real samples to build the curve
        if current_X.shape[0] > max_samples - 5:
            print(
                "current_X is bigger than max samples - 5 so dropping some currently running parameters"
            )
            current_X = current_X[:(max_samples - 5), :]
    if len(sample_y.shape) != 1:
        raise ValueError("Sample y must be a 1 dimensional array")

    if sample_X.shape[0] != sample_y.shape[0]:
        raise ValueError(
            "Sample X and sample y must be the same size {} {}".format(
                sample_X.shape[0], sample_y.shape[0]
            )
        )

    if X_bounds is not None and sample_X.shape[1] != len(X_bounds):
        raise ValueError(
            "Bounds must be the same length as Sample X's second dimension"
        )

    # gaussian process takes a long time to train, so if there's more than max_samples
    # we need to sample from it
    if sample_X.shape[0] > max_samples:
        sample_indices = np.random.randint(sample_X.shape[0], size=max_samples)
        X = sample_X[sample_indices]
        y = sample_y[sample_indices]
    else:
        X = sample_X
        y = sample_y
    gp, y_mean, y_stddev = fit_normalized_gaussian_process(X, y, nu=nu)
    if current_X is not None:
        # if we have some hyperparameters running, we pretend that they return
        # the prediction of the function we've fit
        X = np.append(X, current_X, axis=0)
        current_y_fantasy = (gp.predict(current_X) * y_stddev) + y_mean
        y = np.append(y, current_y_fantasy)
        gp, y_mean, y_stddev = fit_normalized_gaussian_process(X, y, nu=nu)
    return gp.predict, y_mean, y_stddev


def filter_weird_values(sample_X, sample_y):
    is_row_finite = ~(np.isnan(sample_X).any(axis=1) | np.isnan(sample_y))
    sample_X = sample_X[is_row_finite, :]
    sample_y = sample_y[is_row_finite]
    return sample_X, sample_y


def next_sample(
    sample_X,
    sample_y,
    X_bounds=None,
    runtimes=None,
    failures=None,
    current_X=None,
    nu=1.5,
    max_samples_for_gp=100,
    improvement=0.01,
    num_points_to_try=1000,
    opt_func="expected_improvement",
    failure_cost=0,
    test_X=None,
):
    """
        Calculates the best next sample to look at via bayesian optimization.
        Check out https://arxiv.org/pdf/1206.2944.pdf
         for explanation of bayesian optimization
        Arguments:
            sample_X - 2d array of already evaluated sets of hyperparameters
            sample_y - 1d array of already evaluated loss function values
            X_bounds - 2d array minimum and maximum values for every dimension of X
            runtimes - vector of length sample_y - should be the time taken to train each model in sample X
            failures - vector of length sample_y - should be True for models where training failed and False where
                training succeeded.  This model will throw out NaNs and Infs so if you want it to avaoid
                failure values for X, use this failure vector.
            current_X - hyperparameters currently being explored
            nu - input to the Matern function, higher numbers make it smoother 0.5, 1.5, 2.5 are good values
             see http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html
            max_samples_for_gp - maximum samples to consider (since algo is O(n^3)) for performance, but also adds some randomness
            improvement - amount of improvement to optimize for -- higher means take more exploratory risks
            num_points_to_try - number of X values to try when looking for value with highest
                        expected probability of improvement
            opt_func - one of {"expected_improvement", "prob_of_improvement"} - whether to optimize expected
                improvement of probability of improvement.  Expected improvement is generally better - may want
                to remove probability of improvement at some point.  (But I think prboability of improvement
                is a little easier to calculate)
            test_X - X values to test when looking for the best values to try
        Returns:
            suggested_X - X vector to try running next
            suggested_X_prob_of_improvement - probability of the X vector beating the current best
            suggested_X_predicted_y - predicted output of the X vector
            test_X - 2d array of length num_points_to_try by num features: tested X values
            y_pred - 1d array of length num_points_to_try: predicted values for test_X
            y_pred_std - 1d array of length num_points_to_try: predicted std deviation for test_X
            e_i - expected improvement
            prob_of_improve 1d array of lenth num_points_to_try: predicted porbability of improvement
            prob_of_failure 1d array of predicted probabilites of failure
            suggested_X_prob_of_failure
            expected_runtime 1d array of expected runtimes
    """
    # Sanity check the data
    sample_X = np.array(sample_X)
    sample_y = np.array(sample_y)
    failures = np.array(failures)
    if test_X is not None:
        test_X = np.array(test_X)
    if len(sample_X.shape) != 2:
        raise ValueError("Sample X must be a 2 dimensional array")

    if len(sample_y.shape) != 1:
        raise ValueError("Sample y must be a 1 dimensional array")

    if sample_X.shape[0] != sample_y.shape[0]:
        raise ValueError("Sample X and y must be same length")

    if test_X is not None:
        # if test_X is set, usually this is for simulation/testing
        if X_bounds is not None:
            raise ValueError("Can't set test_X and X_bounds")

    else:
        # normal case where we randomly sample our test_X
        if X_bounds is None:
            raise ValueError("Must pass in test_X or X_bounds")

    filtered_X, filtered_y = filter_weird_values(sample_X, sample_y)
    # We train our runtime prediction model on *filtered_X* throwing out the sample points with
    # NaN values because they might break our runtime predictor
    runtime_model = None
    if runtimes is not None:
        runtime_filtered_X, runtime_filtered_runtimes = filter_weird_values(
            sample_X, runtimes
        )
        if runtime_filtered_X.shape[0] >= 2:
            runtime_model, runtime_model_mean, runtime_model_stddev = train_runtime_model(
                runtime_filtered_X, runtime_filtered_runtimes, X_bounds
            )
    # We train our failure model on *sample_X*, all the data including NaNs
    # This is *different* than the runtime model.
    failure_model = None
    if failures is not None and sample_X.shape[0] >= 2:
        failure_filtered_X, failure_filtered_y = filter_weird_values(
            sample_X, failures
        )
        if failure_filtered_X.shape[0] >= 2:
            failure_model, failure_model_mean, failure_model_stddev = train_runtime_model(
                failure_filtered_X, failure_filtered_y, X_bounds, model='rbf_svm'#'logistic'
            )
    # we can't run this algothim with less than two sample points, so we'll
    # just return a random point
    if filtered_X.shape[0] < 2:
        if test_X is not None:
            # pick a random row from test_X
            row = np.random.choice(test_X.shape[0])
            X = test_X[row, :]
        else:
            X = random_sample(X_bounds, 1)[0]
        if filtered_X.shape[0] < 1:
            prediction = 0.0
        else:
            prediction = filtered_y[0]
        return X, 1.0, prediction, None, None, None, None, None, None, None

    # build the acquisition function
    gp, y_mean, y_stddev, = train_gaussian_process(
        filtered_X, filtered_y, X_bounds, current_X, nu, max_samples_for_gp
    )
    # Look for the minimum value of our fitted-target-function + (kappa * fitted-target-std_dev)
    if test_X is None:  # this is the usual case
        test_X = random_sample(X_bounds, num_points_to_try)
    y_pred, y_pred_std = gp(test_X, return_std=True)
    if failure_model is None:
        prob_of_failure = np.zeros(len(test_X))
    else:
        prob_of_failure = failure_model(
            test_X
        ) * failure_model_stddev + failure_model_mean
        #print(f"prob_of_failure: {prob_of_failure}")
    k = 2
    a = 2
    prob_of_failure = a * prob_of_failure**k / (a * prob_of_failure**k + (1 - prob_of_failure)**k)
    if runtime_model is None:
        expected_runtime = [0.0] * len(test_X)
    else:
        expected_runtime = runtime_model(
            test_X
        ) * runtime_model_stddev + runtime_model_mean
    # best value of y we've seen so far.  i.e. y*
    min_unnorm_y = np.min(filtered_y)
    # hack for dealing with predicted std of 0
    epsilon = 0.00000001
    if opt_func == "probability_of_improvement":
        # might remove the norm_improvement at some point
        # find best chance of an improvement by "at least norm improvement"
        # so if norm_improvement is zero, we are looking for best chance of any
        # improvment over the best result observerd so far.
        #norm_improvement = improvement / y_stddev
        min_norm_y = (min_unnorm_y - y_mean) / y_stddev - improvement
        distance = (y_pred - min_norm_y)
        std_dev_distance = (y_pred - min_norm_y) / (y_pred_std + epsilon)
        prob_of_improve = sigmoid(-std_dev_distance)
        if failure_cost > 0:
            prob_of_success = 1 - prob_of_failure
            prob_of_improve *= prob_of_success
        best_test_X_index = np.argmax(prob_of_improve)
        e_i = np.zeros_like(prob_of_improve)
    elif opt_func == "expected_improvement":
        min_norm_y = (min_unnorm_y - y_mean) / y_stddev
        Z = -(y_pred - min_norm_y) / (y_pred_std + epsilon)
        prob_of_improve = scipy_stats.norm.cdf(Z)
        e_i = -(y_pred - min_norm_y) * scipy_stats.norm.cdf(Z) + y_pred_std * scipy_stats.norm.pdf(
            Z
        )
        if failure_cost != 0:
            prob_of_success = 1 - prob_of_failure
            e_i = e_i * prob_of_success / (prob_of_failure * failure_cost + prob_of_success)
            #e_i = e_i * (prob_of_failure < failure_cost)
        best_test_X_index = np.argmax(e_i)
    # TODO: support expected improvement per time by dividing e_i by runtime
    suggested_X = test_X[best_test_X_index]
    suggested_X_prob_of_improvement = prob_of_improve[best_test_X_index]
    suggested_X_predicted_y = y_pred[best_test_X_index] * y_stddev + y_mean
    unnorm_y_pred = y_pred * y_stddev + y_mean
    unnorm_y_pred_std = y_pred_std * y_stddev
    unnorm_e_i = e_i * y_stddev
    suggested_X_prob_of_failure = prob_of_failure[best_test_X_index]
    return (
        suggested_X,
        suggested_X_prob_of_improvement,
        suggested_X_predicted_y,
        test_X,
        unnorm_y_pred,
        unnorm_y_pred_std,
        unnorm_e_i,
        prob_of_improve,
        prob_of_failure,
        suggested_X_prob_of_failure,
        expected_runtime,
    )


def target(x):
    return np.exp(-(x - 2) ** 2) + np.exp(-(x - 6) ** 2 / 10) + 1 / (x ** 2 + 1)


def test():
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    from time import sleep
    
    def function(X):
        X = X.copy()
        X[0] = 1 - X[0]
        if np.sum(X) <= 1: #np.dot(X, X) <= 1:
            return -np.dot(X,X) #-np.sum(X).item()
        else:
            return float("nan")
            
    X_bounds = [(0.0,1.0), (0.0,1.0)]
    sample_X = []
    sample_y = []
    failures = []
    failure_cost = .5
    
    # generate samples
    print("Generating random samples... ", end='')
    samples = np.zeros((1000,2))
    for i in range(1000):
        print(f"{i:4d}\b\b\b\b", end='')
        X = np.random.random(size=2)
        while np.isnan(function(X)):
            X = np.random.random(size=2)
        samples[i] = X
    print("Done.")
        
    
    n_x0 = 40
    n_x1 = 40
    X_grid_0, X_grid_1 = np.meshgrid(np.linspace(0,1,n_x0), np.linspace(0,1,n_x1))
    X_grid = np.stack((X_grid_0, X_grid_1), axis=-1)
    X_grid_flat = X_grid.reshape(-1,2)
    
    # plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.show(block = False)
    
    #for i in range(50):
    cost = 0
    while True:
        sample_X_array = np.array(sample_X) if len(sample_X) > 0 else np.zeros((0,0))
        sample = next_sample(
            sample_X = sample_X_array,
            sample_y = sample_y,
            #X_bounds = X_bounds,
            test_X = samples,
            failures = failures,
            #failure_cost = failure_cost,
            opt_func = "probability_of_improvement"
        )
        next_X = sample[0]
        next_prob_fail = sample[9]
        del sample
        #next_X = np.random.random(size=2)
        next_y = function(next_X)

        ax.clear()
        ax.scatter(samples[...,0], samples[...,1], color='black')
        
        if len(failures) - sum(failures) >= 2:
            grid = next_sample(
                sample_X = sample_X_array,
                sample_y = sample_y,
                failures = failures,
                failure_cost = failure_cost,
                test_X = X_grid_flat       
            )
            y_pred = grid[4].reshape(n_x0, n_x1)
            prob_fail = grid[8].reshape(n_x0, n_x1)
            del grid
            ax.plot_surface(X_grid_0, X_grid_1, -y_pred, facecolors=cm.coolwarm(prob_fail), alpha=.5)
            #ax.plot_surface(X_grid_0, X_grid_1, prob_fail, facecolors=cm.coolwarm(-y_pred), alpha=.5)  
        
        sample_X.append(next_X)
        sample_y.append(next_y)
        failures.append(np.isnan(next_y))
        min_y = np.nanmin(sample_y)
        cost = cost + (failure_cost if np.isnan(next_y) else 1)
        
        #print(next_y, next_prob_fail, min_y)
        #print(sample_y)
        print(f"[{cost:.1f}]: X = {tuple(next_X)}, y = {next_y if next_y else 0:.4f}, prob_fail = {next_prob_fail if next_prob_fail else 0:.4f}, min_y = {min_y if min_y else 0:.4f}")

        ax.scatter(np.array(sample_X)[...,0], np.array(sample_X)[...,1], -np.array(sample_y), color='red')
        plt.show(block = False)
        if cost >= 40:
            break     
        plt.pause(1)
    
    #y_func = np.zeros((n_x0, n_x1))
    #for i in range(n_x0):
    #    for j in range(n_x1):
    #        y_func[i,j] = function(X_grid[i,j])
    #ax.plot_surface(X_grid_0, X_grid_1, y_pred)#y_pred)#, color=prob_fail)
    #ax.scatter(np.array(sample_X)[...,0], np.array(sample_X)[...,1], np.array(sample_y))
    #plt.show()
    
    input("Press Enter to Exit...")


if __name__ == '__main__':
    test()