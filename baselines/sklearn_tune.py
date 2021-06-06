from pprint import pprint

from sklearn.model_selection import GridSearchCV

from baselines.sklearn_models import (
    SKLEARN_CLASSREG_HYPERS, SKLEARN_CLASSREG_MODELS)
from baselines.utils.baseline_hyper_tuner import BaselineHyperTuner
from baselines.utils.hyper_tuning_utils import parse_string_list

# Hyperparameter search algorithms
HYPER_SEARCH_ALGS = {
    'Grid': GridSearchCV
}


def run_sklearn_hypertuning(
        dataset, wandb_args, args, c, wandb_run):
    if c.sklearn_hyper_search == 'Random':
        raise NotImplementedError

    search_alg = HYPER_SEARCH_ALGS[c.sklearn_hyper_search]
    models = c.sklearn_model

    if models == 'All':
        models_dict = SKLEARN_CLASSREG_MODELS

    else:
        models = parse_string_list(models)
        models_dict = {}
        for model in models:
            try:
                models_dict[model] = SKLEARN_CLASSREG_MODELS[model]
            except KeyError:
                raise NotImplementedError(
                    f'Have not implemented model {c.sklearn_model}')

    print('Running sklearn tuning loop with models:')
    pprint(models_dict)

    baseline_hyper_tuner = BaselineHyperTuner(
        dataset=dataset, wandb_args=wandb_args, args=args, c=c,
        wandb_run=wandb_run, models_dict=models_dict,
        hypers_dict=SKLEARN_CLASSREG_HYPERS,
        search_alg=search_alg, verbose=c.sklearn_verbose,
        n_jobs=c.sklearn_n_jobs)

    baseline_hyper_tuner.run_hypertuning()
