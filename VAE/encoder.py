import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
      Mapping the input to embedding vector in latent space z

      --------
      Parameters:
          in_channels : uint8
              channels of input image. The default is 3
      -------
      Returns:
          z_mean : mu of latent space. shape (B X D)
          z_log_var : var of latent space. shape (B X D)
          size : size of encoding vector which is used for decoder. shape (3,)
    """

    def __init__(self, in_channels):
        super().__init__()
        hidden_dims = [128, 256, 512]
        layers = []
        for h_dim in hidden_dims:
            layers.append(
              nn.Sequential(
                  nn.Conv2d(in_channels, h_dim, 4, stride=2, padding=1),
                  nn.BatchNorm2d(h_dim),
                  nn.LeakyReLU()
              )
            )
            in_channels = h_dim

        self.encoder_layers = nn.ModuleList(layers)

        self.z_mean = nn.Sequential(
            nn.Linear(8*8*512, 200),
            nn.LeakyReLU(0.1)
        )
        self.z_log_var = nn.Sequential(
            nn.Linear(8*8*512, 200),
            nn.LeakyReLU(0.1)
        )
        self.flatten_size = 8 * 8 * 512

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
                z = z_mean + z_sigma * epsilon
                z_sigma = exp(z_log_var * 0.5)
        """
        z_mean, z_log_var = inputs
        epsilon = torch.randn(torch.size(z_mean)[0], torch.size(z_mean)[1])

        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        size = x.size()
        x = x.reshape(-1, self.flatten_size)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling([z_mean, z_log_var])

        return z, z_mean, z_log_var, size