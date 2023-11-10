"""Optuna optimization of hyperparameters.

Implements optuna optimization algorithms for hyperparameter tuning.

Returns an optuna study class.

  Typical usage examples:
    import json
    import run_optuna
    from sklearn import datasets

    
    X, y = datasets.load_iris(return_X_y=True)
    X = X[y != 0, :2]
    y = y[y != 0].astype(float)
    
    optuna_config = json.load(optuna_config_file)
    svc_hyperparam_config = optuna_config["model_hyperparams"]["SVC"]
    tuning_params = optuna_config["global_tuning_params"]
            
    optimization = run_optuna.optimize_hyperparams(
        X, y,
        optimization_direction=tuning_params["optimization_direction"],
        n_trials=tuning_params["n_trials"],
        n_splits=tuning_params["n_splits"],
        train_size=tuning_params["train_size"],
        hyperparam_config=svc_hyperparam_config,
        define_model_params={'model_name': 'SVC'}, 
        training_params={}
    )
"""
import optuna
import numpy as np
import pandas as pd

import define_models
from sklearn.model_selection import train_test_split, KFold

def sample_hyperparams_with_optuna(trial, hyperparam_config: dict):
    '''
    For a given optuna trial, samples hyperparameters according to
    the hyperparameter configuration dictionary (hyperparam_config).

    Optuna uses the Tree-structured Parzen Estimator to sample hyperparameters
    given a probabilistic distribution estimated from the model performance
    of previous trials.
    
    Arguments:
        trial: optuna.trial.Trial object
        
        hyperparam_config: dictionary, whose keys = hyperparameter names, 
            value = config dict with keys according to hparam type ('spec_type')
                - categorical: spec_type, values
                - double: spec_type, min, max, step, log
                - int: spec_type, min, max, step, log
                - uniform: min, max
                - discrete_uniform: min, max, step
                    Values are sampled from 
                    [min, min+step, min+2*step, ... , min+k*step <= max]
                - constant: spec_type, value (unchanging, not sampled)
            Example:
                hyperparam_config = {
                    'C': {'spec_type': 'double',
                          'min': 1e-3, 'max': 1
                    },
                    'random_state': {'spec_type': 'constant',
                                     'value' : 7
                    }
                }
    '''    
    params = {}     
    # Individually sample each hyperparameter in hyperparam_config
    for key, config in hyperparam_config.items():
        spec_type = config['spec_type']
        
        # Extract type-specific parameters
        if 'step' in config:
            step = config['step']
        else:
            if spec_type == 'integer':
                step = 1
            if spec_type == 'double':
                step = None
        if 'log' in config.keys():
            log = config['log']
        else:
            log = None
            
        # Sample hyperparameter acording to spec_type
        if spec_type == 'categorical':
            params[key] = trial.suggest_categorical(key, config['values'])
        elif spec_type == 'double':
            params[key] = trial.suggest_float(key, config['min'], config['max'], step=step, log=log)
        elif spec_type == 'integer':
            params[key] = trial.suggest_int(key, config['min'], config['max'], step=step, log=log)
        elif spec_type == 'uniform':
            params[key] = trial.suggest_uniform(key, config['min'], config['max'])
        elif spec_type == 'discrete_uniform':
            params[key] = trial.suggest_discrete_uniform(key, config['min'], config['max'], config['step'])
        elif spec_type == 'constant':
            params[key] = config['value']
        else:
            raise Exception(f'Hyperparameter type not valid: {key}, {spec_type}')
    return params
        
def objective(data: dict,
              trial = None, 
              params: dict = {}, 
              hyperparam_config: dict = {}, 
              define_model_fxn = define_models.define_models,
              define_model_params: dict = {}, 
              train_model_fxn = lambda model, X, y: model.fit(X,y), 
              training_params: dict = {}, 
              evaluate_model_fxn = lambda model, X, y: model.score(X,y),
              eval_params: dict = {}):
    '''
    Defines objective function for optuna hyperparameter optimization.

    Arguments:
        data: dictionary of data. Must have keys n-splits and one of the following sets:
            - (n_splits = 2): X_train, y_train, X_test, y_test
            - (n_splits > 2): X, y, splitter.
                
        trial: optuna.trial.Trial object
        params: (Optional) If not empty, a new hyperparameter set is sampled.
        hyperparam_config: dictionary, whose keys = hyperparameter names, 
            value = config dict with keys according to hparam type ('spec_type')
            
        define_model_fxn: function that returns a model. Takes in (sampled) parameters
            and additional parameters given by define_model_params
        define_model_params: Dictionary of additional parameters
        
        train_model_fxn: function that returns a model. Takes in model class, training data,
            and additional parameters given by training_params
        training_params: Dictionary of additional parameters
        
        evaluate_model_fxn: function that returns performance metric(s). Takes in 
            model class, testing data, and additional parameters given by eval_params
        eval_params: Dictionary of additional parameters
        
    Returns:
        metric: performance metric(s) of trained data
    '''
    # Sample new hyperparameters if none are passed
    if len(params) == 0:
        if trial is None:
            raise Exception(f'If sampling hyperparameters, trial can not be None')
        if len(hyperparam_config) > 0:
            params = sample_hyperparams_with_optuna(trial, hyperparam_config)
        else:
            raise Exception(f'Hyperparameter config dict empty.')

    # Evaluate performance of model given a set of hyperparameters
    if data['n_splits'] == 2:
        # Define model
        if 'params' not in define_model_params.keys():
            define_model_params['params'] = params
        else:
            define_model_params['params'] = {
                **define_model_params['params'], **params
            }
        model = define_model_fxn(**define_model_params)
        # Train model
        training_params['model'] = model
        training_params['X'] = data['X_train']
        training_params['y'] = data['y_train']
        model = train_model_fxn(**training_params)
        # Evaluate model
        eval_params['model'] = model
        eval_params['X'] = data['X_test']
        eval_params['y'] = data['y_test']
        metric = evaluate_model_fxn(**eval_params)
        return metric
    else:
        metrics = []
        data['splitter']
        for i, train_index, test_index in data['splitter']:
            # Define model
            define_model_params['params'] = params
            model = define_model_fxn(**define_model_params)
            # Train model
            training_params['model'] = model
            training_params['X'] = data['X'][train_index]
            training_params['y'] = data['y'][train_index]
            model = train_model_fxn(**training_params)
            # Evaluate model
            eval_params['model'] = model
            eval_params['X'] = data['X'][test_index]
            eval_params['y'] = data['y'][test_index]
            metric = evaluate_model_fxn(**eval_params)
            metrics.append(metric)
        return  tuple(np.mean(metrics, axis=0))

def fill_objective(data, params, hyperparam_config, define_model_fxn, define_model_params, 
                   train_model_fxn, training_params,evaluate_model_fxn, eval_params):
    '''
    Helper function for optimize_hyperparams(). Wrapper for function objective(). 

    Returns:
        filled_obj: function
    '''
    def filled_obj(trial):
        return objective(data, trial, params, hyperparam_config, define_model_fxn, define_model_params, 
                         train_model_fxn, training_params,evaluate_model_fxn, eval_params)
    return filled_obj

def optimize_hyperparams(X, y,
                         optimization_direction = 'maximize',
                         n_trials = 150,
                         n_splits: int = 5,
                         split_groups = None,
                         train_size = None,
                         params: dict = {}, 
                         hyperparam_config: dict = {}, 
                         define_model_fxn = default_optuna.define_models,
                         define_model_params: dict = {}, 
                         train_model_fxn = default_optuna.train_models, 
                         training_params: dict = {}, 
                         evaluate_model_fxn = default_optuna.eval_models,
                         eval_params: dict = {},
                         neptune_callback = None):
    """
    Initializes optuna study and performs hyperparameter tuning.
    
    Samples hyperparameter pertaining to model using Tree-structured Parzen Estimator.
    Trains model given sampled hyperparameters and calculates optimization metrics on held-out data.
    
    Arguments:
        X, y: Features and labels, respectively. The rows of X must correspond to the same
            samples as y. 

        optimization_direction: string or length 2+ list. Defines if performance metrics returned
            by evaluate_model_fxn should be minimized or maximized.
        n_trials: number of sampled sets of hyperparameters
        n_splits: Number of folds splititng the input data. Must be an integer >= 2. 
        split_groups: array-like. Group labels for samples used for splitting into folds.
        train_size: If n_splits=2, data is split into training and testing sets. 
            train_size is the fraction of data allocated to the training dataset.
            
        params: (Optional) If not empty, a new hyperparameter set is sampled.
        hyperparam_config: dictionary, whose keys = hyperparameter names, 
            value = config dict with keys according to hparam type ('spec_type')
            
        define_model_fxn: function that returns a model. Takes in (sampled) parameters
            and additional parameters given by define_model_params
        define_model_params: Dictionary of additional parameters
        
        train_model_fxn: function that returns a model. Takes in model class, training data,
            and additional parameters given by training_params
        training_params: Dictionary of additional parameters
        
        evaluate_model_fxn: function that returns performance metric(s). Takes in 
            model class, testing data, and additional parameters given by eval_params
        eval_params: Dictionary of additional parameters
        
        neptune_callback: NeptuneCallback class to document optuna trials on NeptuneAI.
            None if NeptuneAI loggig is not desired. 
        
    Returns:
        study: Optuna object
    """
    # Split data into training/testing sets or folds
    data = {'n_splits': n_splits}
    if n_splits == 2:
        if train_size is None:
            raise Exception(f'Train_size can not be None if n_splits=2.')
        # Split data into training/testing sets
        data['X_train'], data['X_test'], data['y_train'], data['y_test'] = train_test_split(
            X, y, train_size=train_size, random_state=7, stratify=split_groups)
    elif n_splits > 2:
        # Split data into folds if train_size is not set
        if train_size is None:
            data['splitter'] = []
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=7).split(X,y)
            for i, (train_index, test_index) in enumerate(splitter):
                data['splitter'].append([i, train_index, test_index])
            data['X'] = X; data['y'] = y
        else:
            raise Exception(f'Train_size can not be used if n_splits!=2. train_size={train_size}') 
    else:
        raise Exception(f'n_splits can not be lesser than 2. n_splits={n_splits}')
    
    # Define objective function
    specified_objective = fill_objective(data, params, hyperparam_config, define_model_fxn, define_model_params, 
                   train_model_fxn, training_params,evaluate_model_fxn, eval_params)

    # Initialize study
    study = optuna.create_study(directions=optimization_direction)
    # Perform Optimization

    study.optimize(specified_objective, n_trials = n_trials, callbacks = neptune_callback)
    
    return study
