import functools
from copy import deepcopy
from pprint import pprint
from tempfile import TemporaryDirectory

import numpy as np
import wandb
from numba import njit
from scipy.stats import rankdata
from sklearn.metrics import (
    make_scorer, r2_score, mean_squared_error, accuracy_score, log_loss)
from sklearn.neighbors import KNeighborsTransformer
from sklearn.pipeline import Pipeline

from baselines.sklearn_models import LARGE_DATASETS, MEDIUM_DATASETS
from baselines.utils.hyper_tuning_utils import (
    add_baseline_random_state, get_label_log_loss_metric)

"""
AUROC computation by William Wu. 15x faster than sklearn.metrics.roc_auc_score
for a reasonably sized array 
(<< 1 million; at 1M it is 1.5x slower than sklearn).
https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/208031
"""


@njit
def _auc(actual, pred_ranks):
    actual = np.asarray(actual)
    pred_ranks = np.asarray(pred_ranks)
    n_pos = np.sum(actual)
    n_neg = len(actual) - n_pos
    return (np.sum(pred_ranks[actual==1]) - n_pos*(n_pos+1)/2) / (n_pos*n_neg)


def auc(actual, predicted):
    # only class 1 preds for classification
    if predicted.shape[-1] == 2:
        predicted = predicted[:, 1]

    # return roc_auc_score(actual, predicted)
    pred_ranks = rankdata(predicted)
    return _auc(actual, pred_ranks)


def wrapped_partial(func, *args, **kwargs):
    """Partial that propagates __name__ and __doc__."""
    # louistiao.me/posts/adding-__name__-and~
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


def unstd_scorer(scorer, sigma):
    def new_scorer(*args, **kwargs):
        return sigma * scorer(*args, **kwargs)
    return new_scorer


# Define scoring metrics for regression/classification
def get_scoring_metrics(sigma, labels=None):
    """Need to redefine to work with AUROC removal."""
    if labels is None:
        log_score = make_scorer(log_loss, needs_proba=True)
        auc_score = make_scorer(auc, needs_proba=True)
    else:
        # explicitly provide labels for dataset where y_true
        # may not always have all labels
        log_score = make_scorer(
            wrapped_partial(log_loss, labels=labels),
            needs_proba=True)
        auc_score = make_scorer(
            wrapped_partial(auc, labels=labels),
            needs_proba=True)

    scoring = {
        'reg': {
            'r2': make_scorer(r2_score),
            'rmse': make_scorer(
                wrapped_partial(mean_squared_error, squared=False)),
            'mse': make_scorer(
                wrapped_partial(mean_squared_error, squared=True)),
            'rmse_unstd': make_scorer(
                unstd_scorer(
                    wrapped_partial(mean_squared_error, squared=False),
                    sigma)),
            'mse_unstd': make_scorer(
                unstd_scorer(
                    wrapped_partial(mean_squared_error, squared=True),
                    sigma**2)),
        },
        'class': {
            'logloss': log_score,
            'accuracy': make_scorer(accuracy_score),
            'auroc': auc_score,
        }
    }
    return scoring


METRIC_NEEDS_PROBA = ['logloss', 'auroc']

# Define "refit": the metric used to select the top performer
REFIT = {
    'reg': 'mse',
    # 'class': 'accuracy',
    'class': 'logloss'
}

# Objective -- must match with the above refit
OBJECTIVE = {
    'reg': 'minimize',
    'class': 'minimize'
    # 'class': 'maximize'
}


class BaselineHyperTuner:
    def __init__(
            self, dataset, wandb_args, args, c, wandb_run,
            models_dict, hypers_dict, search_alg, verbose=1, n_jobs=-1):
        self.dataset = dataset
        self.wandb_args = wandb_args
        self.args = args
        self.c = c
        self.wandb_run = wandb_run
        self.models_dict = models_dict
        self.hypers_dict = hypers_dict
        self.search_alg = search_alg
        self.verbose = verbose
        self.n_jobs = n_jobs

        self.n_cv_splits = min(dataset.n_cv_splits, c.exp_n_runs)
        self.metadata = self.dataset.metadata  # Will already be loaded

        # Parse metadata
        self.D = self.metadata['D']
        self.cat_target_cols = self.metadata['cat_target_cols']
        self.num_target_cols = self.metadata['num_target_cols']
        self.target_cols = self.cat_target_cols + self.num_target_cols

        # Store data for each model evaluated
        self.model_name = None
        self.model_classes = None

        # Store data for each CV split
        self.cv_index = None
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        self.data_arrs = None
        self.non_target_cols = None
        self.X_train = None
        self.X_train_val = None
        self.X_val = None
        self.X_test = None

        # Store data for each target column (i.e. multitarget class/reg)
        self.y_train = None
        self.y_train_val = None
        self.y_test = None
        self.reg_class_mode = None
        self.model_class = None
        self.model_hypers = None
        self.cv_split = None  # Iter: must be defined prior to sklearn tuning
        self.refit = None
        self.target_col = None
        self.scoring = None
        self.fitted_model = None
        self.hyper_run_dict = None
        self.aggregate_hyper_run_dicts = []

        # Needs to be set for poker-hand, which explicitly needs to
        # specify labels for the log loss at evaluation time, due to
        # an abnormally small val set (and rare classes).
        self.labels = None

    def run_hypertuning(self):

        if self.c.exp_batch_size != -1:
            raise Exception(
                f'Batch size {self.c.exp_batch_size} provided for baseline '
                f'hypertuning; invalid. Provide a full batch by not '
                f'specifying --exp_batch_size or providing '
                f'--exp_batch_size=-1.')

        for model_index, (model_name, model_classes) in enumerate(
                self.models_dict.items()):
            print('\n------------------')
            print(f'Running prediction for model {model_name}.')

            # Set class model parameters
            self.model_name = model_name
            self.model_classes = model_classes

            # KNN on large graphs -- precompute
            if (self.model_name == 'KNN' and
                    self.c.data_set in MEDIUM_DATASETS + LARGE_DATASETS):
                self.knn_precompute_graph = True
                assert not self.c.sklearn_val_final_fit
                print('Precomputing KNN Graph.')
            else:
                self.knn_precompute_graph = False

            # New wandb logger for each run
            if model_index > 0:
                self.wandb_run = wandb.init(**self.wandb_args)
                wandb.config.update(self.args)
                self.dataset.reset_cv_splits()
                self.aggregate_hyper_run_dicts = []

            self.add_model_params_to_config()

            for cv_index in range(self.n_cv_splits):
                print('CV Index: ', cv_index)

                # Load from ColumnEncodingDataset
                self.dataset.load_next_cv_split()

                self.cv_index = cv_index

                print(
                    f'Train-test Split {cv_index + 1}/'
                    f'{self.dataset.n_cv_splits}')

                if self.c.exp_n_runs < self.dataset.n_cv_splits:
                    print(
                        f'c.exp_n_runs = {self.c.exp_n_runs}. '
                        f'Stopping at {self.c.exp_n_runs} splits.')

                self.run_cv_split_hypertuning()

            self.aggregate_results_from_splits()
            self.wandb_run.finish()  # TODO: is this fixed now?

    def add_model_params_to_config(self):
        new_dict = {}
        new_dict['target_cols'] = self.target_cols
        new_dict['model_name'] = self.model_name
        new_dict['search_alg'] = self.c.sklearn_hyper_search
        wandb.config.update(new_dict)

    def run_cv_split_hypertuning(self):
        """
        At this point, the CV split dataset has been loaded and we can
        parse it, running hypertuning.
        """
        cv_dataset = self.dataset.cv_dataset
        self.train_indices, self.val_indices, self.test_indices = (
            tuple(cv_dataset['new_train_val_test_indices']))

        self.data_arrs = cv_dataset['data_arrs']
        self.non_target_cols = sorted(
            list(set(range(self.D)) - set(self.target_cols)))

        X = []

        for i, col in enumerate(self.data_arrs):
            if i in self.non_target_cols:
                col = col[:, :-1]

                if self.model_name == 'XGBoost':
                    col = col.astype(np.float32)

                X.append(col)

        X = np.hstack(X)
        self.X_train = X[self.train_indices]
        self.X_train_val = X[self.train_indices + self.val_indices]
        self.X_val = X[self.val_indices]
        self.X_test = X[self.test_indices]

        self.run_class_reg_hyper_tuning()

    def run_class_reg_hyper_tuning(self):
        """Wrapper: runs prediction over each numerical / categorical col."""
        if self.num_target_cols:
            if len(self.num_target_cols) > 1:
                raise NotImplementedError
                # Build y with multiple targets

                # Wrap our predictive model

                # self.run_col_hyper_tuning(
                #     target_col=self.num_target_cols, reg_class_mode='reg',
                #     y=y_multitarget)
            else:
                num_col_index = self.num_target_cols[0]
                self.run_col_hyper_tuning(
                    target_col=num_col_index, reg_class_mode='reg')

        if self.cat_target_cols:
            for class_col_index in self.cat_target_cols:
                self.run_col_hyper_tuning(
                    target_col=class_col_index, reg_class_mode='class')

    def run_col_hyper_tuning(self, target_col, reg_class_mode,
                             y: np.array = None):
        """Column-specific prediction preprocessing."""
        print(f'Running {reg_class_mode} on col {target_col}.')
        self.reg_class_mode = reg_class_mode
        wandb.config.update({'reg_class_mode': self.reg_class_mode})
        self.model_class = self.model_classes[reg_class_mode]
        self.model_hypers = self.hypers_dict[self.model_name][reg_class_mode]

        # For MLP, TabNet, we have different sweep sets based on size of
        # the data.
        if self.model_name in ['MLP', 'TabNet', 'KNN']:
            self.model_hypers = self.model_hypers(
                self.model_name, self.c.data_set)

        if not isinstance(self.model_hypers, list):
            self.model_hypers = [self.model_hypers]

        if self.model_name in ['TabNet', 'DKL']:
            if self.c.exp_device == 'cuda:0':
                device = ['cuda']
            else:
                device = ['cpu']

            for config in self.model_hypers:
                config['device_name'] = device

            if self.model_name == 'TabNet':
                for config in self.model_hypers:
                    cat_dims = self.dataset.cv_dataset['cat_dims']
                    cat_features = self.dataset.metadata['cat_features']

                    filtered_cat_dims = []
                    filtered_cat_features = []

                    # Assure target column is not included
                    for i, cat_feature in enumerate(cat_features):
                        if cat_feature in self.cat_target_cols:
                            continue
                        filtered_cat_dims.append(cat_dims[i])
                        filtered_cat_features.append(cat_feature)

                    config['cat_dims'] = [filtered_cat_dims]
                    config['cat_idxs'] = [filtered_cat_features]

        if y is None:
            # Get column encoding
            y = self.data_arrs[target_col]

            if reg_class_mode == 'reg':
                # Exclude mask token for regression
                y = y[:, 0]
            else:
                y = np.argmax(y, axis=1)

        self.y_train = y[self.train_indices]
        self.y_train_val = y[self.train_indices + self.val_indices]
        self.y_val = y[self.val_indices]
        self.y_test = y[self.test_indices]

        compute_auroc = True

        # Done to avoid a rare
        # case in which y_true and y_pred contain a different
        # number of classes, which confuses sklearn
        # See https://github.com/scikit-learn/scikit-learn/issues/11777
        if self.reg_class_mode == 'class':
            class_labels = np.unique(y)
            num_class_labels = len(class_labels)
            if num_class_labels > 2:
                compute_auroc = False
                print('Disabling AUROC because multiclass.')
                labels = self.labels = np.sort(class_labels)
                if self.model_name == 'TabNet' and self.c.data_set == 'poker-hand':
                    # Give TabNet an explicit labels argument
                    for config in self.model_hypers:
                        config['labels'] = [labels]
            else:
                labels = None
        else:
            labels = None

        # TODO: consider using the sklearn multitarget wrappers
        #  (but may be extra)
        self.target_col = target_col
        sigma = self.dataset.cv_dataset['sigmas'][self.target_col]

        self.scoring = get_scoring_metrics(
            sigma=sigma,
            labels=labels)[reg_class_mode]

        if compute_auroc is False and self.reg_class_mode == 'class':
            del self.scoring['auroc']

        print(self.scoring)

        self.refit = REFIT[reg_class_mode]

        # Add random seed for specific
        self.model_hypers = add_baseline_random_state(
            self.model_hypers,
            seed=self.c.baseline_seed + self.cv_index)

        self.tune_fit_eval_model()

    def tune_fit_eval_model(self):
        kwargs = {}

        # We are using the best config reported in the TabNet paper, so we
        # can just skip tuning and go directly to the final fit.
        if self.c.data_set in LARGE_DATASETS and self.model_name in [
                'TabNet', 'GradientBoosting']:
            print(
                f'Running {self.model_name} on dataset {self.c.data_set} -- skipping to '
                f'final train/eval with their reported/the best config.')
            best_params = self.model_hypers[0]
            best_params = {key: value[0] for key, value in best_params.items()}
            pprint(best_params)
            self.hyper_run_dict = {}
        else:
            cv_results = self.run_class_reg_cv_split()
            cv_results = self.clean_cv_results(cv_results)
            # print('Logging full cv_results: ')
            # print(f'\t {cv_results}')
            print(cv_results.keys())
            best_params = self.log_top_model(cv_results)

            if self.knn_precompute_graph:
                # Remove 'knnmodel' prefix
                best_params = {
                    key.split('__')[1]: value for key, value in
                    best_params.items()}

        self.fit_eval_model(best_params, **kwargs)
        self.log_split_performance()

    def run_class_reg_cv_split(self):
        """Run hyperparameter tuning for a particular model and column."""
        pprint(self.model_hypers)

        if not isinstance(self.model_hypers, list):
            param_grid = [self.model_hypers]
        else:
            param_grid = self.model_hypers

        # n_jobs = -1 uses all possible cores
        cv_split = iter([(self.train_indices, self.val_indices)])

        refit = False
        if refit:
            raise ValueError(
                'This is *not* supported right now. '
                'If we want to enable this, we need to make sure that the '
                'make_scorer() functions give the correct `greater_is_better` '
                'for each score. However, this will flip the sign of the '
                'score, which in turn means that *our* evaluation in `self.log'
                'top_model()` wil fail!!')

        # Lets precompute the distance graph
        # Due to Tom Dupre la Tour - https://scikit-learn.org/dev/auto_example
        # s/neighbors/plot_caching_nearest_neighbors.html
        if self.knn_precompute_graph:
            assert len(param_grid) == 1
            assert param_grid[0]['weights'] == ['distance'], (
                'This precomputation should only be done with distance '
                'weighting.')
            assert param_grid[0]['p'] == [2], (
                'This precomputation should only be done with L2 norm.')
            print(f'Precomputing KNN graph with dataset {self.c.data_set}.')
            tmpdir_path = f'sklearn_graph_cache_'

            graph_model = KNeighborsTransformer(
                n_neighbors=max(param_grid[0]['n_neighbors']))
            knn_model = self.model_class(metric='precomputed')
            param_grid = [{f'knnmodel__{key}': value
                           for key, value in param_grid[0].items()}]

            with TemporaryDirectory(prefix=tmpdir_path) as tmpdir:
                model = Pipeline(
                    steps=[('graph', graph_model),
                           ('knnmodel', knn_model)],
                    memory=tmpdir)
                clf = self.search_alg(
                    estimator=model, param_grid=param_grid, cv=cv_split,
                    scoring=self.scoring, refit=refit,
                    verbose=self.verbose, n_jobs=self.n_jobs)
                x, y, kwargs = self.get_train_data()
                clf.fit(x, y, **kwargs)

                # # At this point, we have fit the CV folds and the graph model
                # # is using the tmpdir to store the graph. Retrieve it to
                # # be used with our final fit as follows.
                # knn_graph = model[0].fit_transform(X=x, y=y)

        # In all other cases, do normal tuning
        else:
            model = self.model_class()

            clf = self.search_alg(
                estimator=model, param_grid=param_grid, cv=cv_split,
                scoring=self.scoring, refit=refit,
                verbose=self.verbose, n_jobs=self.n_jobs)

            x, y, kwargs = self.get_train_data()

            clf.fit(x, y, **kwargs)

        return clf.cv_results_

    def get_train_data(self):
        kwargs = {}
        y = self.y_train_val

        if self.model_name in ['TabNet', 'DKL']:
            y_val = self.y_val

            if self.reg_class_mode == 'reg':
                y = y[:, np.newaxis]
                y_val = y_val[:, np.newaxis]

            kwargs['eval_set'] = [(self.X_val, y_val)]
            kwargs['eval_metric'] = [
                get_label_log_loss_metric(labels=self.labels)]

        return self.X_train_val, y, kwargs

    def get_final_fit_data(self):
        """Data to be used in the final evaluation of the best
        performing hyperparameter configuration, on a particular
        cross-validation split.

        I.e., using this, we do not allow our model to retrain on val rows.
        """
        kwargs = {}
        y = self.y_train

        if self.model_name in ['TabNet', 'DKL']:
            y_val = self.y_val

            if self.reg_class_mode == 'reg':
                y = y[:, np.newaxis]
                y_val = y_val[:, np.newaxis]

            kwargs['eval_set'] = [(self.X_val, y_val)]

        return self.X_train, y, kwargs

    def clean_cv_results(self, cv_results):
        # * Rename all occurrences of 'test' to 'val',
        #       to avoid downstream confusion

        clean_results = {}
        for key, value in cv_results.items():
            if 'test' in key:
                val_key = key.replace('test', 'val')
                clean_results[val_key] = value
            else:
                clean_results[key] = value

        return clean_results

    def log_top_model(self, cv_results):

        # Print and log hyperparameter settings
        print(
            f'Evaluating top performing {self.reg_class_mode} '
            f'model with settings:')

        # This is "mean" just over that one train_val split
        eval_criteria = f'mean_val_{self.refit}'

        if OBJECTIVE[self.reg_class_mode] == 'minimize':
            best_model_index = np.argmin(cv_results[eval_criteria])
        else:
            best_model_index = np.argmax(cv_results[eval_criteria])

        best_hyper_settings = cv_results['params'][best_model_index]

        for hyper_name, hyper_setting in best_hyper_settings.items():
            if hyper_name in ['verbose', 'allow_writing_files']:
                continue

            print(f'{hyper_name}: {hyper_setting}')

        self.hyper_run_dict = dict(cv_index=self.cv_index)
        self.hyper_run_dict['best_hyper_settings'] = best_hyper_settings

        # best val performance
        for metric_name, scorer in self.scoring.items():
            val_metric_name = f'best_val_{metric_name}'
            self.hyper_run_dict[val_metric_name] = cv_results[
                f'mean_val_{metric_name}'][best_model_index]

        return best_hyper_settings

    def fit_eval_model(self, best_params, **kwargs):
        # train best model on train again
        model = self.model_class(**best_params)

        if self.c.sklearn_val_final_fit:
            print('Using val in final fit.')
            x, y, kwargs = self.get_train_data()
            print('x.shape', x.shape)
            print('y.shape', y.shape)
        else:
            print('Not using val in final fit.')
            x, y, kwargs = self.get_final_fit_data()
            print('x.shape', x.shape)
            print('y.shape', y.shape)

        model.fit(x, y, **kwargs)

        y_pred = model.predict(self.X_test)

        if self.reg_class_mode == 'class':
            y_pred_proba = model.predict_proba(self.X_test)
        # log test performance of that model
        for metric_name, scorer in self.scoring.items():
            if metric_name in METRIC_NEEDS_PROBA:
                y = y_pred_proba
            else:
                y = y_pred

            performance = scorer._score_func(self.y_test, y)
            test_metric_name = f'best_test_{metric_name}'
            print(test_metric_name, performance)
            self.hyper_run_dict[test_metric_name] = performance

        return 1

    def log_split_performance(self):
        print(
            f'Logged hyper tuning results for {self.model_name}, '
            f'dataset {self.c.data_set}, '
            f'cv split {self.cv_index}, '
            f'search alg {self.c.sklearn_hyper_search} to wandb.')

        self.aggregate_hyper_run_dicts.append(deepcopy(self.hyper_run_dict))

    def aggregate_results_from_splits(self):
        """For each metric collect mean/std for val/test values."""
        run_dicts = self.aggregate_hyper_run_dicts
        final_dict = {}

        for metric_name, scorer in self.scoring.items():
            val_test = [f'best_val_{metric_name}', f'best_test_{metric_name}']
            for metric_name in val_test:
                values = []
                for cv_index in range(self.cv_index + 1):
                    values.append(run_dicts[cv_index][metric_name])
                final_dict[f'{metric_name}_mean'] = np.mean(values)
                final_dict[f'{metric_name}_stddev'] = np.std(values)

        # Compute standard error from RMSE std deviation metrics
        rmse_metric_prefixes = [
            'best_val_rmse_', 'best_test_rmse_',
            'best_val_rmse_unstd_', 'best_test_rmse_unstd_']
        for rmse_metric_prefix in rmse_metric_prefixes:
            rmse_stddev_metric = f'{rmse_metric_prefix}stddev'
            if rmse_stddev_metric in final_dict.keys():
                rmse_std_error_metric = f'{rmse_metric_prefix}stderr'
                final_dict[rmse_std_error_metric] = (
                    final_dict[rmse_stddev_metric] /
                    np.sqrt(self.cv_index + 1))

        wandb.run.summary.update(final_dict)
