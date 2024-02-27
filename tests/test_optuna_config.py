from supervised_baselines import optuna_config

def test_config_dict_loading():
    lengths = {
        "global_tuning_params": 5,
        "model_hyperparams": 15
    }
    for k,n_items in lengths.items():
        assert n_items == len(optuna_config.config_dict[k]) 
