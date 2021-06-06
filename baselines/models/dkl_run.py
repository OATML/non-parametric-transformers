"""Train loop based on
https://docs.gpytorch.ai/en/v1.2.1/examples/06_PyTorch_NN_Integration_DKL/
Deep_Kernel_Learning_DenseNet_CIFAR_Tutorial.html
"""
import gpytorch
import torch
import tqdm
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import TensorDataset, DataLoader

from baselines.models.dkl_modules import MLP, DKLClassificationModel, \
    DKLRegressionModel


def main(c, dataset):
    tune_dkl(c, dataset, hyper_dict=None)


def build_dataloaders(dataset, batch_size):
    data_dict = dataset.cv_dataset.data_dict
    metadata = dataset.metadata
    D = metadata['D']
    cat_target_cols, num_target_cols = (
        metadata['cat_target_cols'], metadata['num_target_cols'])
    target_cols = list(sorted(cat_target_cols + num_target_cols))
    non_target_cols = sorted(
        list(set(range(D)) - set(target_cols)))
    train_indices, val_indices, test_indices = (
        tuple(data_dict['new_train_val_test_indices']))
    data_arrs = data_dict['data_arrs']
    X = []
    y = None

    for i, col in enumerate(data_arrs):
        if i in non_target_cols:
            col = col[:, :-1]
            X.append(col)
        else:
            col = col[:, :-1]
            y = col

    X = torch.cat(X, dim=-1)
    X_train, X_val, X_test = X[train_indices], X[val_indices], X[test_indices]

    if y.shape[1] > 1:
        dataset_is_classification = True
        num_classes = y.shape[1]
        y = torch.argmax(y.long(), dim=1)
    else:
        dataset_is_classification = False
        num_classes = None
        y = torch.squeeze(y)

    y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]
    train_dataset, val_dataset, test_dataset = (
        TensorDataset(X_train, y_train),
        TensorDataset(X_val, y_val),
        TensorDataset(X_test, y_test))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return (
        (train_loader, val_loader, test_loader), X.shape[1],
        dataset_is_classification, num_classes, X_train)


def get_likelihood(dataset_is_classification, num_features, num_classes=None):
    if dataset_is_classification:
        likelihood = gpytorch.likelihoods.SoftmaxLikelihood(
            num_features=num_features, num_classes=num_classes)
    else:
        noise_prior = None
        if False:
            noise_prior_loc = 0.1
            noise_prior_scale = 0.1
            noise_prior = gpytorch.priors.NormalPrior(
                loc=noise_prior_loc, scale=noise_prior_scale)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=noise_prior)

    return likelihood


def tune_dkl(c, dataset, hyper_dict):
    batch_size = c.exp_batch_size
    dataloaders, input_dims, is_classification, num_classes, X_train = (
        build_dataloaders(dataset=dataset, batch_size=batch_size))
    train_loader, val_loader, test_loader = dataloaders

    # Define some hypers here

    # This is the output of the feature extractor, which is then
    # transformed to grid space (in classification) or
    # is the size of the inducing points (regression)
    # We init the inducing points by projecting a random selection of
    # training points, and running KMeans on that
    num_features = 10
    hidden_layers = [100]
    dropout_prob = 0.1
    n_epochs = 1000
    lr = 0.001
    feature_extractor_weight_decay = 1e-4
    scheduler__milestones = [0.5 * n_epochs, 0.75 * n_epochs]
    scheduler__gamma = 0.1
    n_inducing_points = 1000

    likelihood = get_likelihood(
        dataset_is_classification=is_classification,
        num_features=num_features, num_classes=num_classes)

    # Define some hypers here

    feature_extractor = MLP(
        input_size=input_dims, hidden_layer_sizes=hidden_layers,
        output_size=num_features, dropout_prob=dropout_prob)

    if is_classification:
        model = DKLClassificationModel(feature_extractor, num_dim=num_features)
    else:
        model = DKLRegressionModel(
            feature_extractor, n_inducing_points, batch_size, X_train)

    # If you run this example without CUDA, I hope you like waiting!
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    # Train loop
    optimizer = SGD([
        {'params': model.feature_extractor.parameters(),
         'weight_decay': feature_extractor_weight_decay},
        {'params': model.gp_layer.hyperparameters(), 'lr': lr * 0.01},
        {'params': model.gp_layer.variational_parameters()},
        {'params': likelihood.parameters()},
    ], lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
    scheduler = MultiStepLR(
        optimizer, milestones=scheduler__milestones, gamma=scheduler__gamma)
    mll = gpytorch.mlls.VariationalELBO(
        likelihood, model.gp_layer, num_data=len(train_loader.dataset))

    def train(epoch):
        model.train()
        likelihood.train()

        minibatch_iter = tqdm.tqdm(train_loader, desc=f"(Epoch {epoch}) Minibatch")
        with gpytorch.settings.num_likelihood_samples(8):
            for data, target in minibatch_iter:
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()

                data = data.reshape(data.size(0), -1)

                optimizer.zero_grad()
                output = model(data)
                loss = -mll(output, target)
                loss.backward()
                optimizer.step()
                minibatch_iter.set_postfix(loss=loss.item())

    def test(data_loader, mode):
        model.eval()
        likelihood.eval()

        if is_classification:
            correct = 0
        else:
            mse = torch.zeros(1)
            num_batches = 0

        # This gives us 16 samples from the predictive distribution
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(16):
            for data, target in data_loader:
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()

                data = data.reshape(data.size(0), -1)

                if is_classification:
                    output = likelihood(model(data))
                    pred = output.probs.mean(0)
                    pred = pred.argmax(-1)  # Taking the mean over all of the sample we've drawn
                    correct += pred.eq(target.view_as(pred)).cpu().sum()
                else:
                    preds = model(data)
                    mse += torch.mean((preds.mean - target.cpu()) ** 2)
                    num_batches += 1

        if is_classification:
            print('{} set: Accuracy: {}/{} ({}%)'.format(
                mode,
                correct, len(test_loader.dataset), 100. * correct / float(len(data_loader.dataset))
            ))
        else:
            print('{} set: MSE: {}'.format(mode, mse / num_batches))

    for epoch in range(1, n_epochs + 1):
        with gpytorch.settings.use_toeplitz(False):
            train(epoch)
            test(val_loader, mode='Val')
            test(data_loader=test_loader, mode='Test')
        scheduler.step()
        state_dict = model.state_dict()
        likelihood_state_dict = likelihood.state_dict()
        torch.save({'model': state_dict, 'likelihood': likelihood_state_dict}, 'dkl_cifar_checkpoint.dat')
