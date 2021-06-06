"""Load model, data and corresponding configs. Trigger training."""
import os
import pathlib
import sys

import numpy as np
import torch
import torch.multiprocessing as mp
import wandb

from baselines.sklearn_tune import run_sklearn_hypertuning
from npt.column_encoding_dataset import ColumnEncodingDataset
from npt.configs import build_parser
from npt.distribution import distributed_train_wrapper
from npt.train import Trainer
from npt.utils.model_init_utils import init_model_opt_scaler_from_dataset
from npt.utils.viz_att_maps import viz_att_maps


def main(args):
    """Load model, data, configs, start training."""
    args, wandb_args = setup_args(args)
    run_cv(args=args, wandb_args=wandb_args)


def setup_args(args):
    print('Configuring arguments...')

    if args.exp_azure_sweep:
        print('Removing old logs.')
        os.system('rm -r wandb')

    if args.np_seed == -1:
        args.np_seed = np.random.randint(0, 1000)
    if args.torch_seed == -1:
        args.torch_seed = np.random.randint(0, 1000)
    if args.exp_name is None:
        args.exp_name = f'{wandb.util.generate_id()}'
    if (args.exp_group is None) and (args.exp_n_runs > 1):
        # Assuming you want to do CV, group runs together.
        args.exp_group = f'{wandb.util.generate_id()}'
        print(f"Doing k-FOLD CV. Assigning group name {args.exp_group}.")

    if args.exp_azure_sweep:
        print("Azure sweep run!")
        # Our configs may run oom. That's okay.
        os.environ['WANDB_AGENT_DISABLE_FLAPPING'] = 'true'

    if not isinstance(args.model_augmentation_bert_mask_prob, dict):
        print('Reading dict for model_augmentation_bert_mask_prob.')
        # Well, this is ugly. But I blame it on argparse.
        # There is just no good way to parse dicts as arguments.
        # Good thing, I don't care about code security.
        exec(
            f'args.model_augmentation_bert_mask_prob = '
            f'{args.model_augmentation_bert_mask_prob}')

    if not isinstance(args.model_label_bert_mask_prob, dict):
        print('Reading dict for model_augmentation_bert_mask_prob.')
        exec(
            f'args.model_label_bert_mask_prob = '
            f'{args.model_label_bert_mask_prob}')

    if not args.model_bert_augmentation:
        for value in args.model_augmentation_bert_mask_prob.values():
            assert value == 0
        for value in args.model_label_bert_mask_prob.values():
            assert value == 1

    if (args.model_class == 'sklearn-baselines' and
        args.sklearn_model == 'TabNet' and not args.data_force_reload):
        raise ValueError('For TabNet, user must specify data_force_reload '
                         'to encode data in a TabNet-compatible manner.')

    pathlib.Path(args.wandb_dir).mkdir(parents=True, exist_ok=True)

    # Set seeds
    np.random.seed(args.np_seed)

    # Resolve CUDA device(s)
    if args.exp_use_cuda and torch.cuda.is_available():
        if args.exp_device is not None:
            print(f'Running model with CUDA on device {args.exp_device}.')
            exp_device = args.exp_device
        else:
            print(f'Running model with CUDA')
            exp_device = 'cuda:0'
    else:
        print('Running model on CPU.')
        exp_device = 'cpu'

    args.exp_device = exp_device

    wandb_args = dict(
        project=args.project,
        entity=args.entity,
        dir=args.wandb_dir,
        reinit=True,
        name=args.exp_name,
        group=args.exp_group)

    return args, wandb_args


def run_cv(args, wandb_args):

    if args.mp_distributed:
        wandb_run = None
        c = args
    else:
        wandb_run = wandb.init(**wandb_args)
        args.cv_index = 0
        wandb.config.update(args, allow_val_change=True)
        c = wandb.config

    if c.model_class == 'NPT':
        run_cv_splits(wandb_args, args, c, wandb_run)
    elif c.model_class == 'sklearn-baselines':
        run_sklearn_hypertuning(
            ColumnEncodingDataset(c), wandb_args, args, c, wandb_run)


def run_cv_splits(wandb_args, args, c, wandb_run):

    dataset = ColumnEncodingDataset(c)

    #######################################################################
    # Distributed Setting
    if c.mp_distributed:
        torch.manual_seed(c.torch_seed)

        # Fix from
        # https://github.com/facebookresearch/maskrcnn-benchmark/issues/103
        # torch.multiprocessing.set_sharing_strategy('file_system')

        dataset.load_next_cv_split()
        dataset.dataset_gen = None
        args = {'dataset': dataset, 'c': c, 'wandb_args': wandb_args}
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8888'
        mp.spawn(
            distributed_train_wrapper, nprocs=c.mp_gpus, args=(args,),
            join=True)
        mp.set_start_method('fork')
        return

    starting_cv_index = 0
    total_n_cv_splits = min(dataset.n_cv_splits, c.exp_n_runs)

    # Since we're doing CV by default, model init is in a loop.
    for cv_index in range(starting_cv_index, total_n_cv_splits):
        print(f'CV Index: {cv_index}')

        print(f'Train-test Split {cv_index + 1}/{dataset.n_cv_splits}')

        if c.exp_n_runs < dataset.n_cv_splits:
            print(
                f'c.exp_n_runs = {c.exp_n_runs}. '
                f'Stopping at {c.exp_n_runs} splits.')

        # New wandb logger for each run
        if cv_index > 0:
            wandb_args['name'] = f'{wandb.util.generate_id()}'
            args.exp_name = wandb_args['name']
            args.cv_index = cv_index
            wandb_run = wandb.init(**wandb_args)
            wandb.config.update(args, allow_val_change=True)

        #######################################################################
        # Load New CV Split
        dataset.load_next_cv_split()

        if c.viz_att_maps:
            print('Attempting to visualize attention maps.')
            return viz_att_maps(c, dataset)

        if c.model_class == 'DKL':
            print(f'Running DKL on dataset {c.data_set}.')
            from baselines.models.dkl_run import main
            return main(c, dataset)

        #######################################################################
        # Initialise Model
        model, optimizer, scaler = init_model_opt_scaler_from_dataset(
            dataset=dataset, c=c, device=c.exp_device)

        # if not c.exp_azure_sweep:
        #     wandb.watch(model, log="all", log_freq=10)

        #######################################################################
        # Run training
        trainer = Trainer(
            model=model, optimizer=optimizer, scaler=scaler,
            c=c, wandb_run=wandb_run, cv_index=cv_index, dataset=dataset)
        trainer.train_and_eval()

        wandb_run.finish()


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    main(args)
