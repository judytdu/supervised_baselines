from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, \
    RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, ElasticNet, Lasso, Ridge, \
    BayesianRidge, QuantileRegressor, OrthogonalMatchingPursuit

"""Define baseline machine learning model.
Instantiates sklearn class specified by model_type.

Typical Usage Examples:
    python 
"""     
def define_models(model_name: str, params: dict={}):
    """
    Default function for defining machine learning models.
    Instantiates model specified by model_name.
    
    Arguments:
        model_name: String defining model to initialize
        
    Returns:
        Object with specified class.
    """
    # Regression Models
    if model_name == "SVR":
        return SVR(**params)
    elif model_name == "MLPRegressor":
        return MLPRegressor(**params)
    elif model_name == "ElasticNet":
        return ElasticNet(**params)
    elif model_name == "Lasso":
        return Lasso(**params)
    elif model_name == "Ridge":
        return Ridge(**params)
    elif model_name == "BayesianRidge":
        return BayesianRidge(**params)
    elif model_name == "QuantileRegressor":
        return QuantileRegressor(**params)
    elif model_name == "OrthogonalMatchingPursuit":
        return OrthogonalMatchingPursuit(**params)
    # Classification Models
    elif model_name == "SVC":
        return SVC(probability = True, **params)
    elif model_name == "MLPClassifier":
        return MLPClassifier(**params)
    elif model_name == "LogisticRegression":
        return LogisticRegression(**params)
    elif model_name == "GaussianNB":
        return GaussianNB(**params)
    elif model_name =="GradientBoostingClassifier":
        return GradientBoostingClassifier(**params)
    elif model_name == "AdaBoostClassifier":
        return AdaBoostClassifier(**params)
    elif model_name == "RandomForestClassifier":
        return RandomForestClassifier(**params)
    else:
        raise Exception("Model name not valid.")

def get_model_names(all_regressors: bool=False):
    """Get list of all model names specified by all_regressors.

    Args:
        all_regressors (bool): If True, retrieves all regressor model names.
            Otherwise, retrieves all classifier model names. Defaults to False.
    """
    if all_regressors:
        model_names = [
            "SVR", "MLPRegressor", "ElasticNet", "Lasso", "Ridge",  
            "BayesianRidge", "QuantileRegressor", "OrthogonalMatchingPursuit"
        ]
    else:
        model_names = [
            "SVC", "MLPClassifier", "LogisticRegression", "GaussianNB", 
            "GradientBoostingClassifier", "AdaBoostClassifier",
            "RandomForestClassifier"
        ]
    return model_names
    
def validate_model_name(model_type: str):
    """Checks if model_type valid model name under the define_models() function.

    Args:
        model_type (str): Name of model
    """
    all_valid_types  = get_model_names(all_regressors=True) \
        + get_model_names(all_regressors=False)
    return model_type in all_valid_types