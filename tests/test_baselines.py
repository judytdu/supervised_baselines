import os
from supervised_baselines import baselines

reg = baselines.SupervisedBaselines("all_regressors")
clf = baselines.SupervisedBaselines("all_classifiers")

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


# Break down test objects
del reg, clf