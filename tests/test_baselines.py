from supervised_baselines import baselines

import os

# Initialize regressor, classifier, and svc classes
os.system("source ~/neptune_api_token.sh")
neptune_api_token = os.environ["NEPTUNE_API_TOKEN"]

params = {
    "neptune_project_name": "ai4all-genomics-demo", 
    "neptune_workspace": "drjudydu"
}
reg = baselines.SupervisedBaselines(model_type="all_regressors", params)
clf = baselines.SupervisedBaselines(model_type="all_classifiers", params)

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
class IrisClassifier(SupervisedBaselines):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
                    
    def load_data(self):
        X, y = datasets.load_iris(return_X_y=True)
        self.X = X[y != 0]
        self.y = y[y != 0].astype(float)
        
svc = IrisClassifier(model_type="SVC", **params)
svc.add_neptune_global_metadata("data type", "test_baselines")
svc.load_data()

def test_crossvalidation_split(svc):
    svc.crossvalidation_split(n_splits=2)
    assert len(data) = 7
    
def test_optuna_fit(svc):
    svc.fit()
    assert svc.self.optuna_best_performance > 0

# Test Eval

# Break down test objects
del IrisClassifier, neptune_api_token, params, reg, clf, svc 