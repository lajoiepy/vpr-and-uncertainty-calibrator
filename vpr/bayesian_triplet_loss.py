import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import math


def negative_loglikelihood(muA, muP, muN, varA, varP, varN, margin=0.0):
    muA2 = muA**2
    muP2 = muP**2
    muN2 = muN**2
    varP2 = varP**2
    varN2 = varN**2
    mu = torch.sum(muP2 + varP - muN2 - varN - 2 * muA * (muP - muN), dim=0)
    T1 = varP2 + 2 * muP2 * varP + 2 * (varA + muA2) * (
        varP + muP2) - 2 * muA2 * muP2 - 4 * muA * muP * varP
    T2 = varN2 + 2 * muN2 * varN + 2 * (varA + muA2) * (
        varN + muN2) - 2 * muA2 * muN2 - 4 * muA * muN * varN
    T3 = 4 * muP * muN * varA
    sigma2 = torch.max(torch.sum(2 * T1 + 2 * T2 - 2 * T3, dim=0),
                       torch.tensor(0.0))
    sigma = sigma2**0.5
    mu = torch.nan_to_num(mu)
    sigma = torch.nan_to_num(sigma)
    probs = Normal(loc=mu, scale=sigma + 1e-8).cdf(torch.tensor(margin))
    nll = -torch.log(probs + 1e-8)
    return nll.mean()


def kl_div_gauss(mu_q, var_q, mu_p, var_p):
    D = mu_q.shape[0]
    # kl diverence for isotropic gaussian
    kl = 0.5 * ((var_q / var_p) * D + \
    1.0 / (var_p) * torch.sum(mu_p**2 + mu_q**2 - 2*mu_p*mu_q) - D + \
    D*(torch.log(var_p) - torch.log(var_q)))
    return kl.mean()

def kl_div_gauss_var(var_q, var_p):
    # kl diverence for isotropic gaussian and a fixed mean
    kl = 0.5 * ((var_q / var_p) - 1 + \
    (torch.log(var_p) - torch.log(var_q)))
    return kl.mean()


def kl_div_vMF(mu_q, var_q):
    D = mu_q.shape[0]
    # we are estimating the variance and not kappa in the network.
    # They are propertional
    kappa_q = 1.0 / var_q
    kl = kappa_q - D * torch.log(torch.tensor(2.0))
    
    return kl.mean()


class BayesianTripletLoss(nn.Module):

    def __init__(self, margin, var_prior, kl_scale_factor=1e-6, is_gaussian=True):
        super(BayesianTripletLoss, self).__init__()

        self.margin = margin
        self.var_prior = var_prior
        self.kl_scale_factor = kl_scale_factor
        self.is_gaussian = is_gaussian

    def forward(self, anchor_mean, anchor_var, positive_mean, positive_var,
                negative_mean, negative_var):

        varA = anchor_var
        varP = positive_var
        varN = negative_var

        muA = anchor_mean
        muP = positive_mean
        muN = negative_mean

        # calculate nll
        nll = negative_loglikelihood(muA,
                                     muP,
                                     muN,
                                     varA,
                                     varP,
                                     varN,
                                     margin=self.margin)

        if self.is_gaussian:
            # KL(anchor|| prior) + KL(positive|| prior) + KL(negative|| prior)
            mu_prior = torch.zeros_like(muA, requires_grad=False)
            var_prior = torch.ones_like(varA, requires_grad=False) * self.var_prior

            kl = (kl_div_gauss(muA, varA, mu_prior, var_prior) + \
            kl_div_gauss(muP, varP, mu_prior, var_prior) + \
            kl_div_gauss(muN, varN, mu_prior, var_prior))
        else:
            kl = (kl_div_vMF(muA, varA) + \
                kl_div_vMF(muP, varP) + \
                kl_div_vMF(muN, varN))

        return nll + self.kl_scale_factor * kl


class KLDivergenceLoss(nn.Module):

    def __init__(self, var_prior):
        super(KLDivergenceLoss, self).__init__()

        self.var_prior = var_prior

    def forward(self, var):
        var_prior = torch.ones_like(var, requires_grad=False) * self.var_prior

        kl = kl_div_gauss_var(var, var_prior)

        return kl
