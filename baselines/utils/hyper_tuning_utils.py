from pytorch_tabnet.metrics import Metric
from sklearn.metrics import log_loss


def add_baseline_random_state(hypers_list, seed):
    random_seed_like_names = ['random_state', 'seed', 'random_seed']
    for hypers_dict in hypers_list:
        for key in random_seed_like_names:
            if key in hypers_dict.keys():
                hypers_dict[key] = [seed]

    return hypers_list


def parse_string_list(string):
    if ',' in string:
        string = string.replace('[', '').replace(']', '')
        string = [i.strip() for i in string.split(',')]
    else:
        string = [string]
    return string


def modified_tabnet(TabNetModel):
    # Add max_epochs, patience, batch_size as member variables to TabNet

    attributes = [
        'n_d', 'n_a', 'n_steps', 'gamma', 'cat_idxs', 'cat_dims',
        'cat_emb_dim', 'n_independent', 'n_shared', 'epsilon', 'momentum',
        'lambda_sparse', 'seed', 'clip_value', 'verbose', 'optimizer_fn',
        'optimizer_params', 'scheduler_fn', 'scheduler_params', 'mask_type',
        'input_dim', 'output_dim', 'device_name', 'labels']

    class ModifiedTabNetModel(TabNetModel):
        """"""
        def __init__(
                # need to list all params here, s.t. sklearn is happy
                self,
                n_d='dummy',
                n_a='dummy',
                n_steps='dummy',
                gamma='dummy',
                cat_idxs='dummy',
                cat_dims='dummy',
                cat_emb_dim='dummy',
                n_independent='dummy',
                n_shared='dummy',
                epsilon='dummy',
                momentum='dummy',
                lambda_sparse='dummy',
                seed='dummy',
                clip_value='dummy',
                verbose='dummy',
                optimizer_fn='dummy',
                optimizer_params='dummy',
                scheduler_fn='dummy',
                scheduler_params='dummy',
                mask_type='dummy',
                input_dim='dummy',
                output_dim='dummy',
                device_name='dummy',
                max_epochs='dummy',
                patience='dummy',
                batch_size='dummy',
                virtual_batch_size='dummy',
                eval_metric='dummy',
                labels='dummy'):

            # intercept kwargs and remove injection attributes
            # set injection attributes (if used at init)
            # however, sklearn does not do this!! sklearn inits with
            # default and then uses set_params()
            self.injected_attributes = dict(
                max_epochs=200,
                patience=15,
                batch_size=1024,
                virtual_batch_size=128,
                eval_metric=None)

            for attribute in self.injected_attributes:
                value = eval(attribute)
                if value != 'dummy':
                    self.injected_attributes[attribute] = value
                    setattr(self, attribute, value)
                else:
                    # need to write default value, s.t. parameter is present
                    setattr(
                        self, attribute,
                        self.injected_attributes[attribute])

            # filter out non-dummy, s.t. default initialisation can still work
            pass_on_kwargs = dict()
            for attribute in attributes:
                value = eval(attribute)
                if value != 'dummy' and attribute != 'labels':
                    pass_on_kwargs[attribute] = value
                if value != 'dummy' and attribute == 'labels':
                    print('Passing labels explicitly to TabNet.')
                    setattr(self, attribute, value)

            super().__init__(**pass_on_kwargs)

        def fit(self, *args, **kwargs):

            # inject desired epochs/patience
            # sklearn does not use __init__ to set params
            # 
            injected_attributes = {
                attribute: getattr(self, attribute)
                for attribute in self.injected_attributes}

            kwargs.update(injected_attributes)

            # Need to switch the metric here.
            try:
                kwargs['eval_metric'] = [
                    get_label_log_loss_metric(self.labels)]
                print('Labels were provided to TabNet run. '
                      'Injecting a logloss with explicit labels to '
                      'avoid edge case bugs.')
                # The above deals with cases like the validation set not
                # covering the full set of labels
            except Exception as e:
                print(e)

            return super().fit(*args, **kwargs)

        def predict_proba(self, *args, **kwargs):
            return super().predict_proba(*args, **kwargs).astype('float64')

        def set_params(self, **kwargs):
            """Used in the Sklearn grid search to set parameters
            before fit and evaluation on any cross-validation split."""
            print('Setting TabNet parameters.')
            for var_name, value in kwargs.items():
                if var_name == 'labels':
                    continue

                # setattr(self, param_key, param_value)
                try:
                    exec(f"global previous_val; previous_val = self.{var_name}")
                    if previous_val != value:  # noqa
                        wrn_msg = (
                            f"NPT Hyperparameter Tuning: {var_name} changed "
                            f"from {previous_val} to {value}")  # noqa
                        print(wrn_msg)
                        exec(f"self.{var_name} = value")
                except AttributeError:
                    exec(f"self.{var_name} = value")

            return self

    return ModifiedTabNetModel


def get_label_log_loss_metric(labels):
    class LabelLogLoss(Metric):
        """
        LogLoss with explicitly specified label set, to avoid edge cases
        in which a batch does not contain all different categories (e.g.,
        this commonly happens in the heavily imbalanced poker-hand
        synthetic dataset).

        Code from TabNet
        https://github.com/dreamquark-ai/tabnet/blob/
        5e4e8099335ebddd6b297b16aa40cf0bad145b4a/pytorch_tabnet/metrics.py#L444
        """

        def __init__(self):
            self._name = "labellogloss"
            self._maximize = False

        def __call__(self, y_true, y_score):
            """
            Compute LogLoss of predictions.
            Parameters
            ----------
            y_true : np.ndarray
                Target matrix or vector
            y_score : np.ndarray
                Score matrix or vector
            Returns
            -------
            float
                LogLoss of predictions vs targets.
            """
            return log_loss(y_true, y_score, labels=labels)

    return LabelLogLoss
