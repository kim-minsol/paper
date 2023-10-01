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

    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        size = x.size()
        x = x.reshape(-1, self.flatten_size)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)

        return z_mean, z_log_var, size