import torch.nn as nn


class Decoder(nn.Module):
    """
    Decode the latene vector to image which is generated

    --------
    Parameters:
        size : array
            size of encoding vector before flattened
        latent_dim: uint8
            dimension of latent space
        output_dim: uint8
            channels of output image. The default is 3.
    -------
    Returns:
        out: generated image
    """

    def __init__(self, size, latent_dim, output_dim=3):
        super().__init__()
        self.size = size
        hidden_dims = [512, 256, 128]
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0] * 8 * 8)
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(
              nn.Sequential(
                  nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1], 4, stride=2, padding=1),
                  nn.BatchNorm2d(hidden_dims[i+1]),
                  nn.LeakyReLU(0.1)
              )
            )
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], 4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.LeakyReLU(0.1),
                nn.Conv2d(hidden_dims[-1], output_dim, 1),
                nn.Tanh()
            )
        )
        self.decoder_layers = nn.ModuleList(layers)

    def forward(self, z):
        z = self.decoder_input(z)
        out = z.view(-1, self.size[0], self.size[1], self.size[2])
        for layer in self.decoder_layers:
            out = layer(out)

        return out