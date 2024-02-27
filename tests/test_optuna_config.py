from supervised_baselines import optuna_config

def test_config_dict_loading():
    lengths = {
        "global_tuning_params": "n_trials",
        "model_hyperparams": "SVC"
    }
    for k, v in lengths.items():
        assert v in optuna_config.config_dict[k]
