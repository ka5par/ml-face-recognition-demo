import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def euclidean(x, support_mean):
    """
    Compute euclidean distance between two tensors
    """
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = support_mean.size(0)
    d = x.size(1)
    if d != support_mean.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    support_mean = support_mean.unsqueeze(0).expand(n, m, d)
    x_mu = x - support_mean
    # return torch.pow(x - support_mean, 2).sum(2)
    return (x_mu**2).sum(2)


def mahalanobis(x, support_mean, inv_covmat):
    """
    Calculate mahalanobis distance between two tensors
    """

    # create function to calculate Mahalanobis distance
    n = x.size(0)
    d = x.size(1)

    maha_dists = []
    for class_inv_cov, support_class in zip(inv_covmat, support_mean):
        x_mu = x - support_class.unsqueeze(0).expand(n, d)

        mahal = torch.einsum("ij,jk,ki->i", x_mu, class_inv_cov, x_mu.T)
        maha_dists.append(torch.sqrt(mahal))

    return torch.stack(maha_dists, dim=1)


def cosine(x, support_mean, reg_factor=1):
    """calculate pairwise cosine similarity between two tensors"""
    x = x.detach().cpu().numpy()
    cosine_sim = []
    if reg_factor == 1.0:
        for support_class in support_mean:
            cosine_sim.append(
                -cosine_similarity(x, support_class.unsqueeze(0).cpu().numpy())
            )
        return torch.from_numpy(np.asarray(cosine_sim).squeeze(axis=2).T)
    else:
        for sample in x:
            sample_dist = []
            for support_class in support_mean:
                sample_dist.append(
                    np.dot(sample, support_class)
                    / (
                        reg_factor
                        * ((np.linalg.norm(sample) * np.linalg.norm(support_class)))
                    )
                )
            cosine_sim.append(sample_dist)

        return torch.FloatTensor(cosine_sim)
