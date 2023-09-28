import torch
import torch.nn as nn


class VAE(nn.Module):
    """
    Vanila Varational AutoEncoder.

    --------
    Parameters:
        encoder : pytorch module
            The encoder model which encodes input image to latent vector
        decoder: pytorch module
            The decoder model which decodes latent vector to image
    -------
    Returns:
        out: generated image. shape (B x C x H x W) 
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def sampling(self, inputs):
        """
        Reparameterization trick to sample from N(z_mu, z_var) from N(0,1).

        --------
        Parameters:
            inputs : array [z_mu, z_log_var]
                mean, var of latent space
        -------
        Returns:
            z : Tensor. shape (B x D)
        """
        z_mean, z_log_var = inputs
        epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon  # z = z_mean + z_sigma * epsilon, z_sigma = exp(z_log_var * 0.5)

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.sampling([z_mean, z_log_var])
        out = self.decoder(z)
        return out
