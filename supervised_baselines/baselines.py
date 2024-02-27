import os, logging
from sklearn import datasets

import supervised_baselines.define_models as define_models
import supervised_baselines.optuna_config as optuna_config

"""Main script. Creates class capable of training, monitoring, and logging
results from supervised learning. By default, performs hyperparameter tuning
on folds of the input training data and reports performance metrics on 
NeptuneAI. 
    
Typical Usage Examples:
    example_model = SupervisedBaselines(
        model_type="all_regressors", 
        log_file="supervised_baseline.log",
        optuna_config_dict=optuna_config.config_dict,
    )
    example_model.fit(X, y) 
"""

class SupervisedBaselines():
    """Class defining supervised baseline classifiers or regressors.

    Args:
        model_type (str): Name of model. For valid options, run 
            define_models.get_model_names(all_regressors=True) or 
            define_models.get_model_names(all_regressors=False).
        log_file (str, optional): File name or path to write logging statements. 
            Defaults to "supervised_baseline.log".
        optuna_config (dict, optional):  Optuna configuration file. 
            Defaults to config found at optuna_config.py.
    """
    def __init__(
            self, model_type: str, 
            log_file: str="supervised_baseline.log",
            optuna_config_dict: dict=optuna_config.config_dict,
            ):
        self.X = []
        self.y = []
        self.split_groups = []
        
        self._configure_logger(log_file)
        self._configure_model_and_hyperparams(model_type)
        self._configure_optuna_hyperparams(optuna_config_dict)
    
    def load_data(self):
        """Defines self.X, self.y, and self.split_groups
        """
        pass
    
    def __len__(self):
        return len(self.model_types)

    def __repr__(self):
        return self.model_type

    def _configure_logger(self, log_file):
        # Initialize logger and output format
        self.logger = logging.getLogger(__name__)
        log_fmt = "[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s: %(message)s"
        logging.basicConfig(
            filename=log_file, level=logging.INFO,
            format=log_fmt, datefmt='%H:%M:%S'
        )
    
    def _configure_model_and_hyperparams(self, model_type):
        logging.info(f"Configuring supervised model: {model_type}")
        # Fill self.model_types with all model types to train
        self.model_type = model_type
        if model_type == "all_regressors":
            self.model_types = define_models.get_model_names(all_regressors=True)
        elif model_type == "all_classifiers":
            self.model_types = define_models.get_model_names(all_regressors=False)
        else: 
            if define_models.validate_model_name(model_type):
                self.model_types = [model_type]
            else:
                logging.error(f"Invalid model_type: {model_type}")
        logging.info(f"Total number of model types: {len(self.model_types)}")
    
    def _configure_optuna_hyperparams(self, optuna_config_dict):
        logging.info(f"Configuring Optuna hyperparameters")
        # Extract hypertuning-specific parameters and model-specific parameters
        self.optuna = {}; self.model = {}
        if optuna_config_dict is not None:
            self.global_tuning_params = optuna_config_dict["global_tuning_params"]
            self.model_hyperparams = optuna_config_dict["model_hyperparams"]
        else:
            self.global_tuning_params, self.model_hyperparams = None, None        
        
if __name__ == '__main__':
    # Run typical usage example
    class IrisClassifier(SupervisedBaselines):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
                      
        def load_data(self):
            X, y = datasets.load_iris(return_X_y=True)
            self.X = X[y != 0]
            self.y = y[y != 0].astype(float)
        
        def fit(self):
            super().fit(self.X, self.y)
    
    iris_model = IrisClassifier(
        model_type="all_classifiers"
    )
    iris_model.load_data()
    iris_model.fit()
    print(self.optuna_best_performance)