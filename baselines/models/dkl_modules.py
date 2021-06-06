import math

import gpytorch
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch import nn


# MLP feature extractor
class MLP(nn.Module):
    def __init__(
            self, input_size, hidden_layer_sizes, output_size,
            dropout_prob=None):
        super(MLP, self).__init__()
        fc_layers = []
        all_layer_sizes = [input_size] + hidden_layer_sizes
        for layer_size_idx in range(len(all_layer_sizes) - 1):
            fc_layers.append(
                nn.Linear(all_layer_sizes[layer_size_idx],
                          all_layer_sizes[layer_size_idx + 1]))

        self.fc_layers = nn.ModuleList(fc_layers)
        self.output_layer = nn.Linear(
            hidden_layer_sizes[-1], output_size)

        if dropout_prob is not None:
            self.dropout = torch.nn.Dropout(p=dropout_prob)
        else:
            self.dropout = None

    def forward(self, x):
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            x = F.relu(x)

        if self.dropout is not None:
            x = self.dropout(x)

        output = self.output_layer(x)
        return output


# GP Layer
# Trains one GP per feature, as per the SV-DKL paper
# The outputs of those GPs are mixed in the Softmax Likelihood for classification
class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    def __init__(self, num_dim, grid_bounds=(-10., 10.), grid_size=64):

        if num_dim > 1:
            batch_shape = torch.Size([num_dim])
        else:
            batch_shape = torch.Size([])

        variational_distribution = (
            gpytorch.variational.CholeskyVariationalDistribution(
                num_inducing_points=grid_size, batch_shape=batch_shape))

        # Our base variational strategy is a GridInterpolationVariationalStrategy,
        # which places variational inducing points on a Grid
        # We wrap it with a MultitaskVariationalStrategy so that our output is a vector-valued GP
        variational_strategy = gpytorch.variational.GridInterpolationVariationalStrategy(
                self, grid_size=grid_size, grid_bounds=[grid_bounds],
                variational_distribution=variational_distribution
            )

        if num_dim > 1:
            variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
                variational_strategy, num_tasks=num_dim)

        super().__init__(variational_strategy)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                )
            )
        )

        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


# Stochastic Variational Deep Kernel Learning
# Wilson et al. 2016
# https://arxiv.org/abs/1611.00336
# https://docs.gpytorch.ai/en/v1.2.1/examples/06_PyTorch_NN_Integration_DKL/
# Deep_Kernel_Learning_DenseNet_CIFAR_Tutorial.html
class DKLClassificationModel(gpytorch.Module):
    def __init__(self, feature_extractor, num_dim, grid_bounds=(-10., 10.)):
        super(DKLClassificationModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GaussianProcessLayer(
            num_dim=num_dim, grid_bounds=grid_bounds)
        self.grid_bounds = grid_bounds
        self.num_dim = num_dim

    def forward(self, x):
        features = self.feature_extractor(x)
        features = gpytorch.utils.grid.scale_to_bounds(features, self.grid_bounds[0], self.grid_bounds[1])
        # This next line makes it so that we learn a GP for each feature
        features = features.transpose(-1, -2).unsqueeze(-1)
        res = self.gp_layer(features)
        return res


class DKLInducingPointGP(gpytorch.models.ApproximateGP):
    def __init__(self, n_inducing_points, feature_extractor, batch_size, X_train):
        inducing_points = self.get_inducing_points(X_train, n_inducing_points, feature_extractor, batch_size)
        variational_distribution = (
            gpytorch.variational.CholeskyVariationalDistribution(
                inducing_points.size(0)))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution,
            learn_inducing_locations=True)
        super(DKLInducingPointGP, self).__init__(variational_strategy)
        self.feature_extractor_batch_size = batch_size
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def get_inducing_points(self, X_train, n_inducing_points,
                            feature_extractor, feature_extractor_batch_size):
        if torch.cuda.is_available():
            self.feature_extractor = self.feature_extractor.cuda()

        n_inducing_points = min(X_train.size(0), n_inducing_points)
        n_embeds = min(X_train.size(0), n_inducing_points * 10)
        feature_extractor_embeds = []

        # Input indices to embed
        input_indices = np.random.choice(
            np.arange(X_train.size(0)), size=n_embeds, replace=False)

        with torch.no_grad():
            for i in range(0, n_inducing_points, feature_extractor_batch_size):
                batch_indices = input_indices[i:i+feature_extractor_batch_size]
                feature_extractor_embeds.append(
                    feature_extractor(X_train[batch_indices]))

        feature_extractor_embeds = torch.cat(feature_extractor_embeds).numpy()
        km = KMeans(n_clusters=n_inducing_points)
        km.fit(feature_extractor_embeds)
        if True:
            a = 1
        inducing_points = torch.from_numpy(km.cluster_centers_)
        return inducing_points


class DKLRegressionModel(gpytorch.Module):
    def __init__(self, feature_extractor, n_inducing_points, batch_size, X_train):
        super(DKLRegressionModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = DKLInducingPointGP(n_inducing_points, feature_extractor, batch_size, X_train)

    def forward(self, x):
        features = self.feature_extractor(x)
        res = self.gp_layer(features)
        return res
