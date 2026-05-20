import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
import torch
import copy
from sklearn.manifold import TSNE
import os
from scipy.linalg import sqrtm

from dassl.modeling.ops.style_generators.flow_generator import A3FlowStyleGenerator
from dassl.modeling.ops.style_generators.original_sampler import A0OriginalStyleGenerator

from dassl.modeling.ops.style_alignments.adain_alignment import B0AdaINAlignment


def wasserstein_distance_multivariate(mean1, cov1, mean2, cov2):
    mean_diff = mean1 - mean2
    mean_distance = np.dot(mean_diff, mean_diff)
    sqrt_cov1 = sqrtm(cov1)
    if np.iscomplexobj(sqrt_cov1):
        sqrt_cov1 = sqrt_cov1.real

    cov_sqrt_product = sqrtm(sqrt_cov1 @ cov2 @ sqrt_cov1)
    if np.iscomplexobj(cov_sqrt_product):
        cov_sqrt_product = cov_sqrt_product.real

    cov_term = np.trace(cov1 + cov2 - 2 * cov_sqrt_product)
    wasserstein_distance = np.sqrt(mean_distance + cov_term)
    return wasserstein_distance


class ConstStyle(nn.Module):
    def __init__(self, idx, cfg, eps=1e-6):
        super().__init__()
        self.idx = idx
        self.cfg = cfg
        self.eps = eps
        self.alpha_test = cfg.TRAINER.CONSTSTYLE.ALPHA_TEST

        self.mean = []
        self.std = []
        self.domain = []

        self.const_mean = None
        self.const_cov = None
        self.bayes_cluster = None

        self.style_generator_name = getattr(cfg.TRAINER.CONSTSTYLE, "STYLE_GENERATOR", "A3")
        self.style_alignment_name = getattr(cfg.TRAINER.CONSTSTYLE, "STYLE_ALIGNMENT", "B0")

        self.generator = None
        self.alignment = B0AdaINAlignment(eps=eps)

    def clear_memory(self):
        self.mean = []
        self.std = []
        self.domain = []

    def get_style(self, x):
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        var = (var + self.eps).sqrt()
        mu, var = mu.detach().squeeze().cpu().numpy(), var.detach().squeeze().cpu().numpy()
        return mu, var

    def store_style(self, x, domain):
        mu, var = self.get_style(x)
        self.mean.extend(mu)
        self.std.extend(var)
        self.domain.extend(domain.detach().squeeze().cpu().numpy())

    def cal_mean_std(self):
        mean_list = np.array(self.mean)
        std_list = np.array(self.std)
        stacked_data = np.stack((mean_list, std_list), axis=1)
        reshaped_data = stacked_data.reshape((len(mean_list), -1))

        if self.cfg.CLUSTER == 'domain_label':
            print('Clustering using domain label')
            domain_label = torch.tensor(self.domain)
            unique_domain_label = torch.unique(domain_label)

            means, covs = [], []
            for val in unique_domain_label:
                domain_ele = reshaped_data[domain_label == val]
                domain_bayes_cluster = BayesianGaussianMixture(
                    n_components=1,
                    covariance_type='full',
                    init_params='k-means++',
                    max_iter=200
                )
                domain_bayes_cluster.fit(domain_ele)
                means.append(domain_bayes_cluster.means_[0])
                covs.append(domain_bayes_cluster.covariances_[0])

            cluster_mean = np.mean(means, axis=0)
            cluster_cov = np.mean(covs, axis=0)

        else:
            print('Clustering using GMM')
            print(f'Number of cluster: {self.cfg.NUM_CLUSTERS}')
            self.bayes_cluster = BayesianGaussianMixture(
                n_components=self.cfg.NUM_CLUSTERS,
                covariance_type='full',
                init_params='k-means++',
                max_iter=200
            )
            self.bayes_cluster.fit(reshaped_data)

            labels = self.bayes_cluster.predict(reshaped_data)
            unique_labels, _ = np.unique(labels, return_counts=True)

            cluster_mean = np.mean(
                [self.bayes_cluster.means_[i] for i in range(len(unique_labels))],
                axis=0
            )
            cluster_cov = np.mean(
                [self.bayes_cluster.covariances_[i] for i in range(len(unique_labels))],
                axis=0
            )

        self.const_mean = torch.from_numpy(cluster_mean)
        self.const_cov = torch.from_numpy(cluster_cov)

        style_dim = int(self.const_mean.numel())

#-------------------------#

        generator_name = self.style_generator_name.upper()

        if generator_name == "A0":
            self.generator = A0OriginalStyleGenerator(
                const_mean=self.const_mean,
                const_cov=self.const_cov
            )
        elif generator_name == "A3":
            self.generator = A3FlowStyleGenerator(self.cfg, style_dim=style_dim)
        else:
            raise ValueError(
                f"Unsupported STYLE_GENERATOR={self.style_generator_name}. "
                "Currently implemented: A0, A3."
            )
#-------------------#
        if self.style_alignment_name.upper() != "B0":
            raise ValueError(
                f"Unsupported STYLE_ALIGNMENT={self.style_alignment_name}. "
                "Currently implemented: B0."
            )

    def plot_style_statistics(self, idx, epoch):
        domain_list = np.array(self.domain_list)
        mean_list = copy.copy(self.mean_after)
        std_list = copy.copy(self.std_after)
        mean_list = np.array(mean_list)
        std_list = np.array(std_list)
        stacked_data = np.stack((mean_list, std_list), axis=1)
        reshaped_data = stacked_data.reshape((len(mean_list), -1))

        classes = ['in domain 1', 'in domain 2', 'in domain 3', 'out domain']
        tsne = TSNE(n_components=2, random_state=self.cfg.SEED)
        plot_data = tsne.fit_transform(reshaped_data)

        scatter = plt.scatter(plot_data[:, 0], plot_data[:, 1], c=domain_list)
        plt.legend(handles=scatter.legend_elements()[0], labels=classes)
        save_path = os.path.join(
            f'{self.cfg.OUTPUT_DIR}',
            f'testing-features_after{idx}_epoch{epoch}.png'
        )
        plt.savefig(save_path, dpi=200)
        plt.close()
        plt.cla()
        plt.clf()

    def _get_test_style(self, x):
        mu = x.mean(dim=[2, 3], keepdim=True).detach()
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt().detach()

        const_value = torch.reshape(self.const_mean, (2, -1))
        const_mean = const_value[0].float().to(x.device)
        const_std = const_value[1].float().to(x.device)

        const_mean = torch.reshape(const_mean, (1, const_mean.shape[0], 1, 1))
        const_std = torch.reshape(const_std, (1, const_std.shape[0], 1, 1))

        if self.alpha_test:
            const_std = (1 - self.alpha_test) * const_std + self.alpha_test * sig
            const_mean = (1 - self.alpha_test) * const_mean + self.alpha_test * mu

        return const_mean, const_std

    def forward(self, x, domain, store_feature=False, apply_conststyle=False, is_test=False):
        if store_feature:
            self.store_style(x, domain)

        if (not is_test and np.random.random() > self.cfg.TRAINER.CONSTSTYLE.PROB) or not apply_conststyle:
            return x

        if is_test:
            const_mean, const_std = self._get_test_style(x)
        else:
            if self.generator is None:
                raise RuntimeError(
                    "ConstStyle generator is not initialized. "
                    "cal_mean_std() must be called before applying ConstStyle."
                )
            const_mean, const_std = self.generator(x)

        out = self.alignment(x, const_mean, const_std)
        return out