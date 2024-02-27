import os, logging
import sklearn.datasets

from supervised_baselines import define_models
from supervised_baselines import optuna_config
from supervised_baselines import run_optuna

import neptune.new as neptune
from neptune import management
import neptune.new.integrations.optuna as npt_utils
import neptune.new.integrations.sklearn as npt_performance_utils

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
        neptune_api_token="api_token"
    )
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
    def __init__(self, model_type: str, 
                 neptune_project_name: str, neptune_workspace: str, 
                 neptune_api_token: str = None,
                 log_file: str = "supervised_baseline.log",
                 optuna_config_dict: dict = optuna_config.config_dict):
        self.X = []
        self.y = []
        self.split_groups = None
        self._neptune_api_token = neptune_api_token
        
        self._configure_logger(log_file)
        self._configure_model_and_hyperparams(model_type)
        self._configure_optuna_hyperparams(optuna_config_dict)
        self._configure_neptuneai(
            neptune_project_name, neptune_workspace, neptune_api_token)
    
    def load_data(self):
        """Defines self.X, self.y, self.X_test, self.y_test, self.split_groups
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
            format=log_fmt, datefmt='%H:%M:%S')
    
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
        self.optuna, self.model, self.eval, self.best_metric = {}, {}, {}, {}
        if optuna_config_dict is not None:
            self.global_tuning_params = optuna_config_dict["global_tuning_params"]
            self.model_hyperparams = optuna_config_dict["model_hyperparams"]
        else:
            self.global_tuning_params, self.model_hyperparams = None, None        
        
    def _configure_neptuneai(self, neptune_project_name, neptune_workspace, 
                             neptune_api_token):
        logging.info(
            f"Configuring NeptuneAI: {neptune_workspace}/{neptune_project_name}")
        
        # Optionally load token from environment
        self.neptune_project = f"{neptune_workspace}/{neptune_project_name}"
        self.neptune_logger, self.neptune_callback = None, None        
        if neptune_api_token is None:
            if "NEPTUNE_API_TOKEN" in os.environ:
                self._neptune_api_token = os.environ["NEPTUNE_API_TOKEN"]
            else:
                logging.info("""neptune_api_token not found. 
                             For more info, see neptune.ai/get_my_api_token.""")
        
        # Create project if it does not yet exist
        if self._neptune_api_token is not None:
            neptune_projects = management.get_project_list(
                api_token=self._neptune_api_token)
            
            if self.neptune_project not in neptune_projects:
                management.create_project(
                    workspace=neptune_workspace,
                    name=neptune_project_name,
                    visibility='workspace',
                    api_token=self._neptune_api_token)            
    
    def neptune_initialize_run(self, model_name: str):
        self.neptune_logger = neptune.init_run(
            project=self.neptune_project,
            api_token=self._neptune_api_token,
            name=model_name,
            #source_files=["*.py", "requirements.txt"],
            capture_stdout=True,
            capture_stderr=True,
            capture_hardware_metrics=True)
        self.neptune_callback = npt_utils.NeptuneCallback(self.neptune_logger) 
        
    def neptune_add_token(self, neptune_api_token: str):
        self._neptune_api_token = neptune_api_token
    
    def neptune_del_token(self):
        self._neptune_api_token = None
    
    def neptune_log_metadata(self, model_name: str):
        for k, v in self.global_tuning_params.items():
            self.neptune_logger[k] = v
        for k, v in self.model_hyperparams[model_name].items():
            self.neptune_logger[k] = v
    
    def neptune_add_global_metadata(self, key, value):
        self.global_tuning_params[key] = value
    
    def crossvalidation_split(self):
        """Establish training/validation split that is consistent across models.
        The number of splits is determined by the global config file.

        Returns:
            - data: data dict. Must contain the keys n-splits and the following:
                - (if n_splits = 2): X_train, y_train, X_test, y_test
                - (if n_splits > 2): X, y, splitter.
        """
        n_splits = self.global_tuning_params["n_splits"]
        train_size = self.global_tuning_params["train_size"]
                
        self.data = run_optuna.crossvalidation_split(
            n_splits, train_size, self.X, self.y, self.split_groups
        )

    def fit(self):
        """Perform hyperparameter tuning and final model training
        on all models in self.model_types.
        """
        # Split data into training/validation sets or folds
        self.crossvalidation_split()
        # Perform hyperparameter tuning and fnal training on each model type
        for mt in self.model_types:
            if self._neptune_api_token is not None:
                self.neptune_initialize_run(model_name=mt)
                self.neptune_log_metadata(model_name=mt)
            
            if self.global_tuning_params is not None:
                # Perform optuna hyperparameter tuning
                self.optuna[mt], self.model[mt] = run_optuna.optimize_hyperparams(
                    # Input Data
                    data=self.data,
                    # Sampling, Train/Eval Specifications
                    hyperparam_config=self.model_hyperparams[mt],
                    define_model_fxn=define_models.define_models,
                    define_model_params={"model_name": mt},
                    train_model_fxn=lambda model, X, y: model.fit(X,y), 
                    evaluate_model_fxn=lambda model, X, y: model.score(X,y),
                    # Neptune Logger
                    neptune_callback=[self.neptune_callback],
                    # Trial and Fold Specifications: 
                        # n_trials, n_splits, train_size, 
                        # eval_params, optimization_direction
                    n_trials=self.global_tuning_params["n_trials"],
                    eval_params=self.global_tuning_params["eval_params"],
                    optimization_direction=self.global_tuning_params["optimization_direction"])
                self.best_metric[mt] = self.optuna[mt].best_value
                self.eval[mt] = self.evaluate(self.model[mt])
                
            if self._neptune_api_token is not None:
                self.neptune_logger["performance_metrics"] = self.eval[mt]
                self.neptune_logger.stop()
                self.neptune_callback = None
        self.neptune_del_token()
    
    def evaluate(self, model):
        classifiers = ["all_classifiers"] + define_models.get_model_names(False)
        if self.model_type in classifiers:
            return npt_performance_utils.create_classifier_summary(
                model, self.X, self.X_test, self.y, self.y_test
            )
        else:
            return npt_performance_utils.create_regressor_summary(
                model, self.X, self.X_test, self.y, self.y_test
            )
        
if __name__ == '__main__':
    # Run typical usage example
    os.system("source ~/neptune_api_token.sh")
    class IrisClassifier(SupervisedBaselines):
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
    
    iris_model = IrisClassifier(
        model_type="SVC",
        neptune_project_name="ai4all-genomics-demo", 
        neptune_workspace="drjudydu")
    iris_model.neptune_add_global_metadata("n_trials", 5)
    iris_model.load_data()
    iris_model.fit()
    print(iris_model.best_metric)