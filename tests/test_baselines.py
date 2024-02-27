from supervised_baselines import baselines
from supervised_baselines import run_optuna

import os
import sklearn.datasets

# Initialize regressor, classifier, and svc classes
os.system("source ~/neptune_api_token.sh")
neptune_api_token = os.environ["NEPTUNE_API_TOKEN"]

params = {
    "neptune_project_name": "ai4all-genomics-demo", 
    "neptune_workspace": "drjudydu"
}
reg = baselines.SupervisedBaselines(model_type="all_regressors", **params)
clf = baselines.SupervisedBaselines(model_type="all_classifiers", **params)

# Test __init__() and related helper functions
def test_repr(regressors=reg, classifiers=clf):
    assert (str(regressors) == "all_regressors") & \
        (str(classifiers) == "all_classifiers")
    
def test_len(regressors=reg, classifiers=clf):
    assert len(regressors) + len(classifiers) == 15

def test_configure_logger():
    assert os.path.isfile("supervised_baseline.log")

def test_configure_model_regressors(regressors=reg):
    assert "ElasticNet" in regressors.model_types
    
def test_configure_model_classifiers(classifiers=clf):
    assert "LogisticRegression" in classifiers.model_types

def test_configure_optuna_hyperparams(classifiers=clf):
    assert len(classifiers.model_hyperparams) == 15

# Test Neptune
def test_configure_neptuneai(classifiers=clf):
    assert len(classifiers._neptune_api_token) > 0
    
def test_neptune_del_token(classifiers=clf):
    classifiers.neptune_del_token()
    assert not classifiers._neptune_api_token
    
def test_neptune_add_token(classifiers=clf, token=neptune_api_token):
    classifiers.neptune_add_token(token)
    assert classifiers._neptune_api_token
    
def test_neptune_initialize_run(classifiers=clf):
    classifiers.neptune_initialize_run("test_baselines")
    assert True

# Test Fit
class IrisClassifier(baselines.SupervisedBaselines):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
                    
    def load_data(self):
        # Load Iris dataset
        X, y = sklearn.datasets.load_iris(return_X_y=True)
        X = X[ y!=0 ]
        y = y[ y!= 0 ].astype(float)
        
        # Define training/testing split
        data_split = run_optuna.crossvalidation_split(
            n_splits=2, train_size=0.8, X=X, y=y)
        
        self.X, self.y = data_split["X_train"], data_split["y_train"]
        self.X_test, self.y_test = data_split["X_test"], data_split["y_test"]
        self.split_groups = None
        
svc = IrisClassifier(model_type="SVC", **params)
svc.neptune_add_global_metadata("data type", "test_baselines")
svc.neptune_add_global_metadata("n_trials", 5)
svc.load_data()

def test_crossvalidation_split(svc=svc):
    svc.crossvalidation_split()
    assert "n_splits" in svc.data
    
def test_optuna_fit(svc=svc):
    svc.fit()
    assert len(svc.best_metric) > 0

# Break down test objects
del neptune_api_token, params, reg, clf, svc, IrisClassifier
os.system("rm supervised_baseline.log")

# Test KFold

# Test validation_split arg