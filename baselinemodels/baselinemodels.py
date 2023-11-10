import logging
import json
from sklearn import datasets

import define_models
import run_optuna
import neptune.new as neptune
from neptune import management
from neptune.new.types import File
import neptune.new.integrations.optuna as npt_utils
    """Main script. Creates class capable of training, monitoring, and logging
    results from supervised learning. By default, performs hyperparameter tuning
    on folds of the input training data, and  reports performance metrics on 
    NeptuneAI. 
        
    Typical Usage Examples:
        example_model = SupervisedBaselines(
            model_type="all_regressors, 
            logging_out="supervised_baseline.log",
            optuna_config_file="optuna_config.json",
            neptune_project_name, 
            neptune_workspace, 
            neptune_api_token)
        example_model.fit(X.y) 
    """
class SupervisedBaselines():
    """Class defining supervised baseline classifiers or regressors
    """
    def __init__(
            self, model_type: str, logging_out: str="supervised_baseline.log",
            optuna_config_file: str="optuna_config.json",
            neptune_project_name: str, neptune_workspace: str, 
            neptune_api_token: str):
        # Initialize Logger
        self.logger = getLogger(__name__)
        log_fmt = "[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s: %(message)s"
        logging.basicConfig(
            filename=logging_out, level=logging.INFO,
            format=log_fmt, datefmt='%H:%M:%S'
        )
       
        # Configure model and hyperparams
        logging.info(f"Configuring supervised model: {model_type}")
        self.model_type = model_type
        if model_type in ["all_classifiers", "all_regressors"]:
            self.model_types = define_models.get_model_names(
                model_type == "all_regressors")
        else: 
            if define_models.validate_model_name(model_type):
                self.model_types = [model_type]
            else:
                logging.error(f"Invalid model_type: {model_type}")
        logging.info(f"Total number of model types: {len(self.model_types)}")
        
        # Configure Optuna hyperparams
        logging.info(f"Configuring Optuna hyperparameters: {optuna_config_file}")
        if optuna_config_file is not None:
            optuna_config = json.load(optuna_config_file)
            self.global_tuning_params = optuna_config["global_tuning_params"]
            self.model_hyperparams = optuna_config["model_hyperparams"]
        else:
            self.global_tuning_params, self.model_hyperparams = None, None
            
        # Configure NeptuneAI
        logging.info(
            f"Configuring NeptuneAI: {neptune_workspace}/{neptune_project_name}"
        )
        self.neptune_project_name = neptune_project_name
        self.neptune_workspace = neptune_workspace
        self.__neptune_api_token = neptune_api_token
        self.neptune_logger, self.neptune_callback = None, None
        
        neptune_projects = management.get_project_list(
            api_token=self.__neptune_api_token
        )
        if self.neptune_project_name not in neptune_projects:
            neptune.management.create_project(
                workspace=self.neptune_workspace,
                name=self.neptune_project_name,
                visibility='workspace',
                api_token=self.__neptune_api_token
            )            
        
    def __len__(self):
        return len(self.model_types)

    def __repr__(self):
        return self.model_type
    
    def neptune_initialize_run(self, model_name: str):
        self.neptune_logger = neptune.init_run(
            project=f'{self.neptune_workspace}/{self.neptune_project_name}',
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
    
    def neptune_log_metadata(self, model_name):
        for k, v in self.global_tuning_params.items():
            self.neptune_logger[k] = self.global_tuning_params[v]
        for k, v in self.model_hyperparams[model_name].items():
            self.neptune_logger[k] = self.model_hyperparams[model_name][v]
            
    def fit(X: np.array,  y:np.array, split_groups: np.array=None):
        for mt in model_type():
            self.neptune_initialize_run(model_name=mt)
            self.neptune_log_metadata(model_name=mt)
            
            if self.global_tuning_params is not None:
            # Perform optuna hyperparameter tuning
            self.optuna = run_optuna.optimize_hyperparams(
                # Input Data
                X, y,
                # KFold Grouping
                split_groups = split_groups,
                # Sampling, Train/Eval Specifications
                hyperparam_config = self.model_hyperparams,
                define_model_fxn = define_models.define_models(mt),
                train_model_fxn = lambda model, X, y: model.fit(X,y), 
                evaluate_model_fxn = lambda model, X, y: model.score(X,y),
                # Neptune Logger
                neptune_callback = [self.neptune_callback] 
                # Trial and Fold Specifications: 
                    # n_trials, n_splits, train_size, 
                    # eval_params, optimization_direction
                **self.global_tuning_params
            )
            
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
    
            hyperparam_configs = {'svm-classifier': get_param_spec('svm-classifier')}
            #hyperparam_configs = get_all_param_specs(all_regressors=True)
    
        def fit(self):
            super().fit(self.X, self.y)
    
    iris_model = IrisClassifier()
    iris_model.fit()