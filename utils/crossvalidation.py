from surprise import accuracy, SVD
from surprise.model_selection import KFold, RepeatedKFold, ShuffleSplit, LeaveOneOut
import pandas as pd

def configure_validation(iterator, data, algo, n_splits=3, metric="rmse"):
    i = iterator(n_splits)
    _algo = algo()
    metric_results = []
    for trainset, testset in i.split(data):
        # train and test algorithm.
        _algo.fit(trainset)
        predictions = _algo.test(testset)
        # Compute and print Root Mean Squared Error
        metric_results.append( accuracy[metric](predictions, verbose=True))
    print(metric_results) 
    return metric_results

metrics = ["rmse", "mse", "mae"]


def get_validation_accuracy(iterator, data, algo, n_splits=3):
    results = pd.DataFrame()
    for metric in metrics:
        results[metric] = configure_validation(iterator,data, algo, n_splits, metric )
    return results

iterators = ["KFold", ]

def perform_cross_validation(algo, data, algo_name, dataset_name, cv=5, measures=["RMSE", "MSE", "MAE"]):
    results = cross_validate(algo(), data, measures=measures, cv=cv, verbose=True)
    results_2 = configure_validation()
    print(results)
    return {
        "Algorithm": algo_name,
        "Dataset": dataset_name,
        "RMSE": results['test_rmse'].mean(),
        "MSE": results['test_mse'].mean(),
        "MAE": results['test_mae'].mean(),
        "FitTime":sum(results["fit_time"]) / len(results["fit_time"]),
        "TestTime":sum(results["test_time"]) / len(results["test_time"])
    }

results_list = []