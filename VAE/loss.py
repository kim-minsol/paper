import torch
from torch.nn import functional as F


def loss_function(x, recon, mu, log_var):
    """
    Loss function of VAE. used Binary Cross Entrophy and KL Divergence term
    Can compute BCE with sampling.
    KLD term calculates the difference between two gaussians

    --------
    Parameters:
        x : image. shape (B x C x H x W)
          input image
        recon : image. shape (B x C x H x W)
          reconstructure of input image x
        mu :
          mean of latent space z
        log_var :
          variance of latent space z
    -------
    Returns:
        total loss  : reconstructure loss + KL Divergence
    """
    recon_loss = F.mse_loss(recon.view(-1), x.view(-1))
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

    return recon_loss + kld_loss