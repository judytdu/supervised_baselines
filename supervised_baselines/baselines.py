import os, logging
from sklearn import datasets

import supervised_baselines.define_models as define_models
import supervised_baselines.optuna_config as optuna_config

from neptune import management
import neptune.new as neptune
import neptune.new.integrations.optuna as npt_utils

"""Main script. Creates class capable of training, monitoring, and logging
results from supervised learning. By default, performs hyperparameter tuning
on folds of the input training data, and reports performance metrics on 
NeptuneAI. 
    
Typical Usage Examples:
    example_model = SupervisedBaselines(
        model_type="all_regressors, 
        log_file="supervised_baseline.log",
        optuna_config_dict=optuna_config.config_dict,
        neptune_project_name="project_name", 
        neptune_workspace="neptuneai_username", 
        neptune_api_token="api_token")
    example_model.fit(X, y) 
"""

class SupervisedBaselines():
    """Class defining supervised baseline classifiers or regressors.

    Args:
        model_type (str): Name of model. For valid options, run 
            define_models.get_model_names(all_regressors=True) or 
            define_models.get_model_names(all_regressors=False).
        neptune_project_name (str): Name of NeptuneAI project.
        neptune_workspace (str): Name of NeptuneAI workspace. Often a username.
        neptune_api_token (str, optional): Neptune API tokens can be passed 
            directly or set as an environment variable. If neptune_api_token 
            is None and NEPTUNE_API_TOKEN is not an environment variable, no 
            results are logged to NeptuneAI. Defaults to None. 
            For more info, see https://app.neptune.ai/get_my_api_token.
        log_file (str, optional): File name or path to write logging statements. 
            Defaults to "supervised_baseline.log".
        optuna_config (dict, optional):  Optuna configuration file. 
            Defaults to config found at optuna_config.py.
    """
    def __init__(
            self, model_type: str, 
            neptune_project_name: str, neptune_workspace: str, 
            neptune_api_token: str=None,
            log_file: str="supervised_baseline.log",
            optuna_config_dict: dict=optuna_config.config_dict,
            ):
        self.X = []
        self.y = []
        self.split_groups = []
        
        self._configure_logger(log_file)
        self._configure_model_and_hyperparams(model_type)
        self._configure_optuna_hyperparams(optuna_config_dict)
        self._configure_neptuneai(
            neptune_project_name, neptune_workspace, neptune_api_token)
    
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
        
        
    def _configure_neptuneai(self, neptune_project_name, neptune_workspace, 
                                  neptune_api_token):
        logging.info(
            f"Configuring NeptuneAI: {neptune_workspace}/{neptune_project_name}"
        )
        # Optionally load token from environment
        self.neptune_project = f"{neptune_workspace}/{neptune_project_name}"
        self.neptune_logger, self.neptune_callback = None, None        
        if neptune_api_token is None:
            if "NEPTUNE_API_TOKEN" in os.environ:
                neptune_api_token = os.environ["NEPTUNE_API_TOKEN"]
            else:
                logging.info("""neptune_api_token not found. 
                             For more info, see neptune.ai/get_my_api_token.""")
        self.__neptune_api_token = neptune_api_token
        
        # Create project if it does not yet exist
        if self.__neptune_api_token is not None:
            neptune_projects = management.get_project_list(
                api_token=self.__neptune_api_token
            )
            if self.neptune_project not in neptune_projects:
                management.create_project(
                    workspace=neptune_workspace,
                    name=neptune_project_name,
                    visibility='workspace',
                    api_token=self.__neptune_api_token
                )            
    
    def neptune_initialize_run(self, model_name: str):
        self.neptune_logger = neptune.init_run(
            project=self.neptune_project,
            api_token=self.__neptune_api_token,
            name=model_name,
            #source_files=["*.py", "requirements.txt"],
            capture_stdout=True,
            capture_stderr=True,
            capture_hardware_metrics=True
        )
        self.neptune_callback = npt_utils.NeptuneCallback(self.neptune_logger) 
        
    def neptune_add_token(self, neptune_api_token: str):
        self.__neptune_api_token = neptune_api_token
    
    def neptune_del_token(self):
        del self.__neptune_api_token
    
    def neptune_log_metadata(self, model_name: str):
        for k, v in self.global_tuning_params.items():
            self.neptune_logger[k] = v
        for k, v in self.model_hyperparams[model_name].items():
            self.neptune_logger[k] = v
            
    def fit(self):
        for mt in self.model_types:
            if self.__neptune_api_token is not None:
                self.neptune_initialize_run(model_name=mt)
                self.neptune_log_metadata(model_name=mt)
            
            if self.global_tuning_params is not None:
                # Perform optuna hyperparameter tuning
                self.optuna[mt], self.model[mt] = run_optuna.optimize_hyperparams(
                    # Input Data
                    self.X, self.y,
                    # KFold Grouping
                    split_groups = split_groups,
                    # Sampling, Train/Eval Specifications
                    hyperparam_config = self.model_hyperparams[mt],
                    define_model_fxn = define_models.define_models,
                    define_model_params = {"model_name": mt},
                    train_model_fxn = lambda model, X, y: model.fit(X,y), 
                    evaluate_model_fxn = lambda model, X, y: model.score(X,y),
                    # Neptune Logger
                    neptune_callback = [self.neptune_callback],
                    # Trial and Fold Specifications: 
                        # n_trials, n_splits, train_size, 
                        # eval_params, optimization_direction
                    **self.global_tuning_params
                )
                self.optuna_best_performance[mt] = self.optuna[mt].best_value
            if self.__neptune_api_token is not None:
                self.neptune_logger.stop()
                self.neptune_callback = None
        self.neptune_del_token()
        
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
        model_type="all_classifiers",
        neptune_project_name ="supervised-baselines-example", 
        neptune_workspace="drjudydu")
    iris_model.load_data()
    iris_model.fit()
    print(self.optuna_best_performance)