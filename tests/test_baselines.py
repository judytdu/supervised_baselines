import os
from supervised_baselines import baselines

# Initialize regressor and classifier classes
os.system("source ~/neptune_api_token.sh")
params = {
    model_type = "all_regressors",
    neptune_project_name ="supervised-baselines-example", 
    neptune_workspace="drjudydu"
}
reg = baselines.SupervisedBaselines(**params)
params["model_type"] = "all_classifiers"
clf = baselines.SupervisedBaselines(**params)

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
    assert len(classifiers.__neptune_api_token) > 0
    
def test_neptune_del_token(classifiers=clf):
    classifiers.neptune_del_token()
    assert len(classifiers.__neptune_api_token) == 0
    
def test_neptune_add_token(classifiers=clf):
    classifiers.neptune_add_token(os.environ["NEPTUNE_API_TOKEN"])
    assert len(classifiers.__neptune_api_token) > 0
    
def test_neptune_initialize_run(classifiers=clf):
    classifiers.neptune_initialize_run("test_baselines")
    assert True

# Test Fit

# Test Eval

# Break down test objects
del params, reg, clf