from functools import partial

from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMRegressor, LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestRegressor, RandomForestClassifier)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor

from baselines.utils.hyper_tuning_utils import modified_tabnet

SKLEARN_CLASSREG_MODELS = {
    'XGBoost': {
        'reg': XGBRegressor,
        'class': partial(XGBClassifier, use_label_encoder=False)},
    'GradientBoosting': {
        'reg': GradientBoostingRegressor,
        'class': GradientBoostingClassifier},
    'RandomForest': {
        'reg': RandomForestRegressor,
        'class': RandomForestClassifier},
    'CatBoost': {
        'reg': CatBoostRegressor,
        'class': CatBoostClassifier},
    'MLP': {
        'reg': MLPRegressor,
        'class': MLPClassifier},
    'LightGBM': {
        'reg': LGBMRegressor,
        'class': LGBMClassifier
    },
    'TabNet': {
        'reg': modified_tabnet(TabNetRegressor),
        'class': modified_tabnet(TabNetClassifier),
    },
    'KNN': {
        'reg': KNeighborsRegressor,
        'class': KNeighborsClassifier
    },
}

# 24 options
RANDOM_FOREST_CLASS_HYPERS = {
    'random_state': [],  # Added from config
    'criterion': ['gini', 'entropy'],
    'n_estimators': [50, 100, 500, 1000],
    'max_features': ['auto', 'sqrt', 'log2'],
}

# 24 options
RANDOM_FOREST_REG_HYPERS = {
    'random_state': [],  # Added from config
    'criterion': ['mae', 'mse'],
    'n_estimators': [50, 100, 500, 1000],
    'max_features': ['auto', 'sqrt', 'log2'],
}

# 48 options
GRADIENT_BOOSTING_HYPERS = {
    'random_state': [],  # Added from config
    'learning_rate': [1e-3, 1e-2, 0.1, 0.3],
    'max_depth': [3, 5, 10],
    'n_estimators': [50, 100, 500, 1000],
}

# 48 options
XGB_HYPERS = {
    'random_state': [],  # Added from config
    'learning_rate': [1e-3, 1e-2, 0.1, 0.3],
    'max_depth': [3, 5, 10],
    'n_estimators': [50, 100, 500, 1000],
}

# 48 options
LIGHTGBM_HYPERS = {
    'random_state': [],  # Added from config
    'learning_rate': [1e-3, 1e-2, 0.1, 0.3],
    'max_depth': [3, 5, 10],
    'n_estimators': [50, 100, 500, 1000],
}

# 48 options
CATBOOST_HYPERS = {
    'random_seed': [],  # Added from config
    # 'verbose': [False],  # Has particularly annoying logging
    # 'allow_writing_files': [False],
    'learning_rate': [1e-3, 1e-2, 0.1, 0.3],
    'max_depth': [3, 5, 10],
    'n_estimators': [50, 100, 500, 1000],
}

# 480 options
# No random_state/random_seed params for KNN
SMALL_DATA_KNN_HYPERS = {
    'n_neighbors': [2, 5, 7, 10, 20, 30, 40, 50, 75, 100],
    'weights': ['uniform', 'distance'],
    'algorithm': ['ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [10, 30, 50, 100],
    'p': [1, 2]
}

# 40 options
# We precompute the KNN graph for medium/large data.
# No need to use an approximate search e.g., ball_tree or kd_tree.
# We will precompute all the distances, then use brute-force search.
MEDIUM_DATA_KNN_HYPERS = {
    'n_neighbors': [2, 5, 7, 10, 25, 50, 100, 250, 500, 1000],
    'weights': ['distance'],  # Don't bother trying uniform weights
    'leaf_size': [10, 30, 50, 100],
    'p': [2]
}

# 40 options
# We can't afford doing a very high number of n_neighbors for the
# largest datasets, and anyway, we probably don't want to.
LARGE_DATA_KNN_HYPERS = {
    'n_neighbors': [2, 3, 4, 5, 7, 10, 12, 15, 20, 25],
    'weights': ['distance'],  # Don't bother trying uniform weights
    'leaf_size': [10, 30, 50, 100],
    'p': [2]
}

# 11340 options
SMALL_DATA_MLP_HYPERS = {
    'random_state': [],  # Added from config
    'hidden_layer_sizes': [
        (25,), (50,), (100,), (250,), (500,),
        (25, 25), (50, 50), (100, 100), (250, 250), (500, 500),
        (25, 25, 25), (50, 50, 50), (100, 100, 100),
        (250, 250, 250), (500, 500, 500)],
    'alpha': [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],  # L2 reg
    'batch_size': [32, 64, 128, 256],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'learning_rate_init': [
        1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
    'early_stopping': [True]
}

# 270 options
MEDIUM_LARGE_MLP_HYPERS = {
    'random_state': [],  # Added from config
    'hidden_layer_sizes': [
        (25, 25, 25), (100, 100, 100), (500, 500, 500)],
    'alpha': [0, 1e-4, 1e-2],  # L2 reg
    'batch_size': [128, 256],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'learning_rate_init': [
        1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    'early_stopping': [True]
}

# 6 options
HIGGS_MLP_HYPERS = {
    'random_state': [],  # Added from config
    'hidden_layer_sizes': [
        (500, 500, 500)],
    'alpha': [0],  # L2 reg
    'batch_size': [512, 1024],
    'learning_rate': ['constant'],
    'learning_rate_init': [
        1e-4, 1e-3, 1e-2],
    'early_stopping': [True]
}

# Models for which we have a different hyperparameter config sweep
# based on the size of the data.
SPLIT_HYPER_MODEL_NAMES = ['MLP', 'TabNet', 'KNN']
SMALL_DATASETS = [
        'boston-housing', 'yacht', 'concrete', 'breast-cancer',
        'energy-efficiency']
MEDIUM_DATASETS = [
    'poker-hand', 'protein', 'sarcos', 'income', 'kick']
LARGE_DATASETS = [
    'forest-cover', 'higgs']


def dataset_to_hypers(model_name, dataset_name):
    if model_name not in SPLIT_HYPER_MODEL_NAMES:
        raise ValueError(
            f'Attempted to find a hyperparameter configuration for dataset '
            f'{dataset_name} and model {model_name}. No call to '
            f'`dataset_to_hypers` needed.')

    if model_name == 'MLP':
        if dataset_name == 'higgs':
            print('Using Higgs MLP hypers.')
            return HIGGS_MLP_HYPERS
        elif dataset_name in SMALL_DATASETS:
            print(f'Using small data MLP hypers on dataset {dataset_name}.')
            return SMALL_DATA_MLP_HYPERS
        elif dataset_name in MEDIUM_DATASETS or dataset_name in LARGE_DATASETS:
            print(
                f'Using med/large data MLP hypers on dataset {dataset_name}.')
            return MEDIUM_LARGE_MLP_HYPERS

    if model_name == 'TabNet':
        if dataset_name in SMALL_DATASETS:
            print(f'Sweeping over small data TabNet '
                  f'hypers on dataset {dataset_name}.')
            return SMALL_DATA_TABNET_HYPERS
        elif dataset_name in MEDIUM_DATASETS:
            print(f'Sweeping over medium data TabNet '
                  f'hypers on dataset {dataset_name}.')
            return MEDIUM_DATA_TABNET_HYPERS
        elif dataset_name in LARGE_DATASETS:
            print(
                f'Using TabNet reported config for large '
                f'dataset {dataset_name}.')
            return LARGE_DATA_TABNET_HYPERS[dataset_name]

    if model_name == 'KNN':
        if dataset_name in SMALL_DATASETS:
            print(f'Using small data KNN hypers on dataset {dataset_name}.')
            return SMALL_DATA_KNN_HYPERS
        elif dataset_name in MEDIUM_DATASETS:
            print(
                f'Using med data KNN hypers on dataset {dataset_name}.')
            return MEDIUM_DATA_KNN_HYPERS
        elif dataset_name in LARGE_DATASETS:
            print(
                f'Using large data KNN hypers on dataset {dataset_name}.')
            return LARGE_DATA_KNN_HYPERS

    raise NotImplementedError(
        f'No dataset {dataset_name} known to sklearn baselines.')


SMALL_DATA_TABNET_HYPERS = {
    'seed': [],
    'n_d': [8, 16],
    'n_a': [8, 16],
    'lambda_sparse': [0.0001],
    'batch_size': [16, 32, 64, 128],
    'virtual_batch_size': [4, 8, 16],
    'momentum': [0.02],
    'n_steps': [3],
    'gamma': [1.3],
    'optimizer_params': [dict(lr=2e-2)],
    'scheduler_params': [dict(step_size=10, gamma=0.95)],
    'max_epochs': [3000],
    'device_name': [],
    'patience': [100],  # Default is 15
}

# For forest-cover/higgs, use the TabNet config for the respective dataset
# (meaning we can skip Grid Search), unlimited patience, and fix epochs
# to that reported in their paper.
LARGE_DATA_TABNET_HYPERS = {
    # Train size: 406707
    # Batch size: 16384
    # Steps per epoch: 24
    # Reported steps: 130,000
    # Resultant epochs: ~5400
    'forest-cover': {
        'seed': [],
         'n_d': [64],
         'n_a': [64],
         'lambda_sparse': [0.0001],
         'batch_size': [4096],
         'virtual_batch_size': [512],
         'momentum': [0.7],
         'n_steps': [5],
         'gamma': [1.5],
         'optimizer_params': [dict(lr=2e-2)],
         'scheduler_params': [dict(step_size=500, gamma=0.95)],
         'max_epochs': [5400],
         'device_name': [],
         'patience': [100000]
     },
    # Train size: 9187500
    # Batch size: 8192
    # Steps per epoch: 1121
    # Reported steps: 370,000
    # Resultant epochs: ~330
    'higgs': {
        'seed': [],
        'n_d': [96],
        'n_a': [32],
        'lambda_sparse': [0.000001],
        'batch_size': [8192],
        'virtual_batch_size': [256],
        'momentum': [0.9],
        'n_steps': [8],
        'gamma': [2],
        'optimizer_params': [dict(lr=0.025)],
        'scheduler_params': [dict(step_size=10000, gamma=0.9)],
        'max_epochs': [330],
        'device_name': [],
        'patience': [100000]
    }
}

# For medium data, sweep over all TabNet listed configurations
# with higher patience (15 -> 100) and higher max number of epochs (3000)
MEDIUM_DATA_TABNET_HYPERS = [
    # forest-cover
    {
        'seed': [],
        'n_d': [64],
        'n_a': [64],
        'lambda_sparse': [0.0001],
        'batch_size': [16384],
        'virtual_batch_size': [512],
        'momentum': [0.7],
        'n_steps': [5],
        'gamma': [1.5],
        'optimizer_params': [dict(lr=2e-2)],
        'scheduler_params': [dict(step_size=500, gamma=0.95)],
        'max_epochs': [3000],
        'device_name': [],
        'patience': [100],  # Default is 15
    },
    # higgs
    {
        'seed': [],
        'n_d': [96],
        'n_a': [32],
        'lambda_sparse': [0.000001],
        'batch_size': [8192],
        'virtual_batch_size': [256],
        'momentum': [0.9],
        'n_steps': [8],
        'gamma': [2],
        'optimizer_params': [dict(lr=0.025)],
        'scheduler_params': [dict(step_size=10000, gamma=0.9)],
        'max_epochs': [3000],
        'device_name': [],
        'patience': [100],  # Default is 15
    },
    # income model
    {
         'seed': [],
         'n_d': [16],
         'n_a': [16],
         'lambda_sparse': [0.0001],
         'batch_size': [4096],
         'virtual_batch_size': [128],
         'momentum': [0.98],
         'n_steps': [5],
         'gamma': [1.5],
         'optimizer_params': [dict(lr=2e-2)],
         'scheduler_params': [dict(step_size=2500, gamma=0.4)],
         'max_epochs': [3000],
         'device_name': [],
         'patience': [100],  # Default is 15
    },
    # sarcos model
    {
        'seed': [],
        'n_d': [128],
        'n_a': [128],
        'lambda_sparse': [0.0001],
        'batch_size': [4096],
        'virtual_batch_size': [128],
        'momentum': [0.8],
        'n_steps': [5],
        'gamma': [1.5],
        'optimizer_params': [dict(lr=2e-2)],
        'scheduler_params': [dict(step_size=8000, gamma=0.9)],
        'max_epochs': [3000],
        'device_name': [],
        'patience': [100],  # Default is 15
    },
    # mushroom model
    {
        'seed': [],
        'n_d': [8],
        'n_a': [8],
        'lambda_sparse': [0.001],
        'batch_size': [2048],
        'virtual_batch_size': [128],
        'momentum': [0.9],
        'n_steps': [3],
        'gamma': [1.5],
        'optimizer_params': [dict(lr=2e-2)],
        'scheduler_params': [dict(step_size=2500, gamma=0.4)],
        'max_epochs': [3000],
        'device_name': [],
        'patience': [100],  # Default is 15
    },
    # poker hand model
    {
        'seed': [],
        'n_d': [16],
        'n_a': [16],
        'lambda_sparse': [0.000001],
        'batch_size': [4096],
        'virtual_batch_size': [1024],
        'momentum': [0.95],
        'n_steps': [4],
        'gamma': [1.5],
        'optimizer_params': [dict(lr=0.01)],
        'scheduler_params': [dict(step_size=500, gamma=0.95)],
        'max_epochs': [3000],
        'device_name': [],
        'patience': [100],  # Default is 15
    },
    # default model with higher max epochs/patience
    {
        'seed': [],
        'max_epochs': [3000],
        'device_name': [],
        'patience': [100],  # Default is 15
    },
]


def tabnet_add_eval_metric(class_reg_mode, hypers):
    if class_reg_mode == 'reg':
        eval_metric = 'mse'
    elif class_reg_mode == 'class':
        eval_metric = 'logloss'
    else:
        raise NotImplementedError

    if isinstance(hypers, list):
        for d in hypers:
            d['eval_metric'] = [[eval_metric]]
    elif isinstance(hypers, dict):
        hypers['eval_metric'] = [[eval_metric]]

    return hypers


# May be different hypers amongst classification versus regression (e.g. SVM)
SKLEARN_CLASSREG_HYPERS = {
    'RandomForest': {
        'reg': RANDOM_FOREST_REG_HYPERS,
        'class': RANDOM_FOREST_CLASS_HYPERS},
    'GradientBoosting': {
        'reg': GRADIENT_BOOSTING_HYPERS,
        'class': GRADIENT_BOOSTING_HYPERS},
    'XGBoost': {
        'reg': XGB_HYPERS,
        'class': {**XGB_HYPERS, **dict(eval_metric=['logloss'])}},
    'CatBoost': {
        'reg': CATBOOST_HYPERS,
        'class': CATBOOST_HYPERS},
    'MLP': {
        'reg': dataset_to_hypers,
        'class': dataset_to_hypers},
    'TabNet': {
        'reg': lambda model, dataset: tabnet_add_eval_metric(
            'reg', dataset_to_hypers(model, dataset)),
        'class': lambda model, dataset: tabnet_add_eval_metric(
            'class', dataset_to_hypers(model, dataset))},
    'LightGBM': {
        'reg': LIGHTGBM_HYPERS,
        'class': LIGHTGBM_HYPERS},
    'KNN': {
        'reg': dataset_to_hypers,
        'class': dataset_to_hypers},
}
