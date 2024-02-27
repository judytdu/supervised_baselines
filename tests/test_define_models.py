import pytest
from supervised_baselines import define_models
from sklearn.svm import SVC

# Test define_models()
def test_define_models_invalid():
    with pytest.raises(Exception):
        define_models.define_models("SVM")

def test_define_models_valid():
    assert define_models.define_models("SVC").C == SVC().C

# Test get_model_names()
def test_get_model_names_regressors():
    names = define_models.get_model_names(all_regressors=True)
    assert "ElasticNet" in names
    
def test_get_model_names_classifiers():
    names = define_models.get_model_names(all_regressors=False)
    assert "LogisticRegression" in names

# Test validate_model_name()
def test_validate_model_name_invalid():
    assert not define_models.validate_model_name("SVM")

def test_validate_model_names_valid():
    assert define_models.validate_model_name("SVC")
    