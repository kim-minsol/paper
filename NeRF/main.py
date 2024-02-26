import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import NeRF


def accumulate_transmittance(alphas):
    accumulated = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated.shape[0], 1), device=alphas.device), accumulated[:,:-1]), dim=-1)


def render_rays(nerf, rays_o, rays_d, device='cuda', near=0, far=1, N_samples=192):
    z_val = torch.linspace(near, far, N_samples, device=device).expand(rays_o.shape[0], N_samples)
    # stratified sampling
    mid = (z_val[:, 1:] + z_val[:, :-1]) * .5
    lower = torch.cat([z_val[...,:1], mid], dim=-1)
    upper = torch.cat([mid, z_val[...,-1:]], dim=-1)
    t_rand = torch.rand(z_val.shape, device=device)
    t = lower + (upper - lower) * t_rand  # reparameterization trick / (batch_size, N_samples)

    x = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * t.unsqueeze(2) # (batch_size, N_samples, 3)
    # x = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]

    rays_d = rays_d.unsqueeze(1).expand(rays_d.shape[0], N_samples, 3) # (batch_size, 3) -> (batch_size, N_samples, 3)

    colors, sigma = nerf(x.reshape(-1, 3), rays_d.reshape(-1, 3))
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    delta = torch.cat((z_val[:, 1:] - z_val[:, :-1], torch.tensor([1e10], device=device).expand(rays_d.shape[0], 1)), -1) # (batch_size, N_samples)
    alpha = 1 - torch.exp(-sigma * delta)
    weights = accumulate_transmittance(1 - alpha). unsqueeze(2) * alpha.unsqueeze(2)
    c = torch.sum(weights * colors, dim=1)
    weight_sum = weights.sum(-1).sum(-1)

    return c + 1 - weight_sum.unsqueeze(-1)


def test(hn, hf, dataset, model, device='cuda', chunk_size=10, img_index=0, N_samples=192, H=400, W=400):
    """
    Args:
        hn: near plane distance
        hf: far plane distance
        dataset: dataset to render
        model: neural radiance model
        chunk_size (int, optional): chunk size for memory efficiency. Defaults to 10.
        img_index (int, optional): image index to render. Defaults to 0.
        N_samples (int, optional): number of bins for density estimation. Defaults to 192.
        H (int, optional): image height. Defaults to 400.
        W (int, optional): image width. Defaults to 400.

    Returns:
        None: None
    """
    ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]

    data = []   # list of regenerated pixel values
    with torch.no_grad():
        for i in range(int(np.ceil(H / chunk_size))):   # iterate over chunks
            # Get chunk of rays
            ray_origins_ = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
            ray_directions_ = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
            regenerated_px_values = render_rays(model, ray_origins_, ray_directions_, device=device, hn=hn, hf=hf, N_samples=N_samples)
            data.append(regenerated_px_values)
        img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)

    plt.figure()
    plt.imshow(img)
    plt.savefig(f'novel_views/img_{img_index}.png', bbox_inches='tight')
    plt.close()


def train(nerf, optim, schedular, data_loader, device='cuda', near=0, far=1, epochs=int(1e5), N_samples=192, H=400, W=400):
    training_loss = []
    for _ in tqdm(range(epochs)):
        for batch in data_loader:
            rays_o = batch[:,:3].to(device)
            rays_d = batch[:,3:6].to(device)
            gt_pixels = batch[:,6:].to(device)

            recon_pixels = render_rays(nerf, rays_o, rays_d, device, near, far, N_samples)
            loss = ((gt_pixels - recon_pixels) ** 2).sum()

            optim.zero_grad()
            loss.backward()
            optim.step()
            training_loss.append(loss.item())
        schedular.step()

        for img_index in range(200):
            test(near, far, testing_dataset, device=device, img_index=img_index, N_samples=N_samples, H=H, W=W)

    return training_loss


if __name__=='__main__':
    train_dataset = torch.from_numpy(np.load('~~~~', allow_pickle=True))
    testing_dataset = torch.from_numpy(np.load('~~~~', allow_pickle=True))

    device = "cuda"

    model = NeRF().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=5e-4)
    schedular = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[2, 4, 8], gamma=0.5)
    data_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    train(model, optim, schedular, data_loader, device=device, near=2, far=6, epochs=16, N_samples=192, H=400, W=400)