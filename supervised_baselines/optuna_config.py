config_dict = {
    "global_tuning_params": {
        "n_trials": 50,
        "n_splits": 2,
        "train_size": 0.8,
        "split_groups": None,
        "eval_params": {},
        "optimization_direction": ["maximize"]
    },
    # n_splits: Number of folds splititng the input data. Must be an int >= 2. 
    # split_groups: array-like. Group labels for samples for fold splitting.
    # train_size: If n_splits=2, data is split into training and testing sets. 
        # train_size is the fraction of data allocated to the training dataset.
            

    "model_hyperparams": {
        # Regression Models
        "SVR": {
            "C": {
                "spec_type": "double",
                "min": 1e-3, "max": 1, "log": True
            },
            "kernel": {
                "spec_type": "categorical",
                "values": ["linear", "poly", "rbf", "sigmoid"]
            },
            "degree": {"spec_type": "integer",
                        "min": 3, "max": 6
            }
        },

        "MLPRegressor": {
            "hidden_layer_sizes": {"spec_type": "integer",
                      "min": 100, "max": 1000,
            },
            "alpha": {"spec_type": "double",
                      "min": 5e-6, "max": 5e-4, "log": True
            },
            "random_state": {"spec_type": "constant",
                             "value" : 7
            },
            "learning_rate": {"spec_type": "constant",
                             "value" : "adaptive"
            }
        },
        
        "ElasticNet": {
            "alpha": {"spec_type": "double",
                      "min": 1e-3, "max": 5, "log": True
            },
            "l1_ratio": {"spec_type": "uniform",
                      "min": 0, "max": 1
            },
            "random_state": {"spec_type": "constant",
                             "value" : 7
            }
        },

        "Lasso": {
            "alpha": {"spec_type": "double",
                      "min": 1e-3, "max": 5, "log": True
            },
            "random_state": {"spec_type": "constant",
                             "value" : 7
            }
        },

        "Ridge": {
            "alpha": {"spec_type": "double",
                      "min": 1e-3, "max": 5, "log": True
            },
            "random_state": {"spec_type": "constant",
                             "value" : 7
            }
        },

        "BayesianRidge": {
            "alpha_1": {"spec_type": "double",
                      "min": 1e-9, "max": 1e-3, "log": True
            },
            "alpha_2": {"spec_type": "double",
                      "min": 1e-9, "max": 1e-3, "log": True
            },
            "compute_score": {"spec_type": "constant",
                             "value" : True
            }
        },
        
        "QuantileRegressor": {
            "quantile": {"spec_type": "double",
                      "min": 0.375, "max": 0.625
            },
        }, 
        
        "OrthogonalMatchingPursuit" : {
            "n_nonzero_coefs": {"spec_type": "integer",
                      "min": 1, "max": 100
            }
        },

        # Classification Models
        "SVC": {
            "C": {"spec_type": "double",
                    "min": 1e-3, "max": 1, "log": True
            },
            "kernel": {"spec_type": "categorical",
                        "values": ["linear", "poly", "rbf", "sigmoid"]
            },
            "degree": {"spec_type": "integer",
                        "min": 3, "max": 6
            },
            "random_state": {"spec_type": "constant",
                                "value" : 7
            },
            "class_weight": {"spec_type": "constant",
                                "value" : "balanced"
            }
        },
        
        "MLPClassifier": {
            "hidden_layer_sizes": {"spec_type": "integer",
                      "min": 100, "max": 1000,
            },
            "alpha": {"spec_type": "double",
                      "min": 5e-6, "max": 5e-4, "log": True
            },
            "random_state": {"spec_type": "constant",
                             "value" : 7
            },
            "learning_rate": {"spec_type": "constant",
                             "value" : "adaptive"
            }
        }, 

        "LogisticRegression": {
            "C": {"spec_type": "double",
                      "min": 1, "max": 1000, "log": True
            },
            "l1_ratio": {"spec_type": "uniform",
                      "min": 0, "max": 1
            },
            "penalty": {"spec_type": "constant",
                        "value" : "elasticnet"
            },
            "solver": {"spec_type": "constant",
                       "value" : "saga"
            },
           "max_iter": {"spec_type": "constant",
                        "value" : 1000
            },
            "random_state": {"spec_type": "constant",
                             "value" : 7
            }
        }, 
        
        "GaussianNB": {
            "var_smoothing": {"spec_type": "double",
                      "min": 1e-10, "max": 1e-6, "log": True
            }
        }, 

        "GradientBoostingClassifier": {
            "max_depth": {"spec_type": "integer",
                    "min": 25, "max": 200
            },
            "n_estimators": {"spec_type": "integer",
                    "min": 300, "max": 600
            },
            "min_samples_split": {"spec_type": "integer",
                    "min": 2, "max": 10
            },
            "min_impurity_decrease": {"spec_type": "double",
                    "min": 0.0, "max": 0.25
            },
            "min_samples_leaf": {"spec_type": "integer",
                    "min": 5, "max": 25
            },
            "random_state": {"spec_type": "constant",
                            "value" : 7
            }
        }, 
        ## Unaltered default params: loss="deviance", learning_rate=0.1, 
            # subsample=1.0, criterion="friedman_mse", max_leaf_nodes=None,
            # min_weight_fraction_leaf=0.0, min_impurity_split=None,ccp_alpha=0.0
            # validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, 
            # min_weight_fraction_leaf=0.0
        "AdaBoostClassifier": {
            "n_estimators": {"spec_type": "integer",
                      "min": 300, "max": 600
            },
            "learning_rate": {"spec_type": "double",
                      "min": 0, "max": 1e4
            },
            "random_state": {"spec_type": "constant",
                      "value" : 7
            }
        },
        
        "RandomForestClassifier": {
            "max_depth": {"spec_type": "integer",
                      "min": 25, "max": 200
            },
            "n_estimators": {"spec_type": "integer",
                      "min": 300, "max": 600
            },
            "min_samples_split": {"spec_type": "integer",
                      "min": 2, "max": 10
            },
            "min_impurity_decrease": {"spec_type": "double",
                      "min": 0.0, "max": 0.25
            },
            "min_samples_leaf": {"spec_type": "integer",
                      "min": 5, "max": 25
            },
            "random_state": {"spec_type": "constant",
                             "value" : 7
            }
        }
    }
}
