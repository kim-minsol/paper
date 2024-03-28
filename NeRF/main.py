import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import NeRF


def exists(x):
    return x is not None


def accumulate_transmittance(alphas):
    accumulated = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated.shape[0], 1), device=alphas.device), accumulated[:,:-1]), dim=-1)


# Hierarchical sampling
def sample_pdf(bins, weights, N_samples):
    # pdf -> cdf / bins, weights -> (batch, N_bins - 1)
    weights = weights + 1e-5 # preprocessing nan
    pdf = weights / torch.sum(weights, -1, keepdim=True) # normalize
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1) # (batch, len(bins))

    # uniform samples
    u = torch.linspace(0., 1., steps=N_samples, device=cdf.device)
    u = u.expand(list(cdf.shape[:-1]) + [N_samples]) # (batch, N_samples)

    # inverse transform sampling
    u = u.contiguous() # memory contiguous
    idxs = torch.searchsorted(cdf, u, right=True) # u가 cdf 내에서 정렬되기 위해 들어가는 위치 -> idxs에 저장
    below = torch.max(torch.zeros_like(idxs-1), idxs-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(idxs), idxs)
    idxs_g = torch.stack([below, above], -1) # (batch, N_samples, 2)

    matched_shape = [idxs_g.shape[0], idxs_g.shape[1], cdf.shape[-1]] # (batch, N_samples, len(bins))
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, idxs_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, idxs_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[...,0]) / denom
    samples = bins_g[...,0] + t * (bins_g[...,1] - bins_g[...,0])

    return samples


def render_rays(model_coarse, rays_o, rays_d, device='cuda', near=0, far=1, model_fine=None, N_samples=192, N_importance=5):
    z_vals = torch.linspace(near, far, N_samples, device=device).expand(rays_o.shape[0], N_samples)

    # stratified sampling
    mid = (z_vals[:, 1:] + z_vals[:, :-1]) * .5
    lower = torch.cat([z_vals[...,:1], mid], dim=-1)
    upper = torch.cat([mid, z_vals[...,-1:]], dim=-1)
    t_rand = torch.rand(z_vals.shape, device=device)
    z_vals = lower + (upper - lower) * t_rand  # reparameterization trick / (batch_size, N_samples)

    # x = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * t.unsqueeze(2)
    x = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # (batch_size, N_samples, 3)

    viewdirs = rays_d.unsqueeze(1).expand(rays_d.shape[0], N_samples, 3) # (batch_size, 3) -> (batch_size, N_samples, 3)

    # model_coarse outputs
    colors, sigma = model_coarse(x.reshape(-1, 3), viewdirs.reshape(-1, 3))
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    # calculate alpha, weights
    delta = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.tensor([1e10], device=device).expand(rays_d.shape[0], 1)), -1) # (batch_size, N_samples)
    alpha = 1 - torch.exp(-sigma * delta)

    # weights = accumulate_transmittance(1 - alpha). unsqueeze(2) * alpha.unsqueeze(2)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    c = torch.sum(weights[...,None] * colors, -2) # rgb map
    weight_sum = torch.sum(weights, -1) # accumulated opacity


    # fine network
    if exists(model_fine):
        z_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_mid, weights[...,1:-1], N_importance)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # (N_rays, N_samples + N_importance, 3)
        viewdirs = rays_d.unsqueeze(1).expand(rays_d.shape[0], N_samples + N_importance, 3) # (batch_size, 3) -> (batch_size, N_samples, 3)

        # model_fine outputs
        colors, sigma = model_fine(pts.reshape(-1, 3), viewdirs.reshape(-1, 3))
        colors = colors.reshape(pts.shape)
        sigma = sigma.reshape(pts.shape[:-1])

        # calculate alpha, weights
        delta = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.tensor([1e10], device=device).expand(rays_d.shape[0], 1)), -1) # (batch_size, N_samples)
        alpha = 1 - torch.exp(-sigma * delta)
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1.-alpha + 1e-10], -1), -1)[:, :-1] # accumulate_transmittance
        c = torch.sum(weights[...,None] * colors, -2) # rgb map
        weight_sum = torch.sum(weights, -1) # accumulated opacity

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


def train(model_coarse, optim, schedular, data_loader, device='cuda', near=0, far=1, epochs=int(1e5), model_fine=None, N_samples=192, H=400, W=400):
    training_loss = []
    for _ in tqdm(range(epochs)):
        for batch in data_loader:
            rays_o = batch[:,:3].to(device)
            rays_d = batch[:,3:6].to(device)
            gt_pixels = batch[:,6:].to(device)

            recon_pixels = render_rays(model_coarse, rays_o, rays_d, device, near, far, model_fine, N_samples)
            loss = ((gt_pixels - recon_pixels) ** 2).sum()

            optim.zero_grad()
            loss.backward()
            optim.step()
            training_loss.append(loss.item())
        schedular.step()

        for img_index in range(200):
            test(near, far, test_dataset, device=device, img_index=img_index, N_samples=N_samples, H=H, W=W)

    return training_loss


if __name__ == '__main__':
    train_dataset = torch.from_numpy(np.load('~~~~', allow_pickle=True))
    test_dataset = torch.from_numpy(np.load('~~~~', allow_pickle=True))

    device = "cuda"

    # create NeRF
    model_coarse = NeRF().to(device)
    model_fine = NeRF().to(device)

    grad_vars = list(model_coarse.parameters()) + list(model_fine.parameters())
    optim = torch.optim.Adam(grad_vars, lr=5e-4, betas=(0.9, 0.999))
    schedular = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[2, 4, 8], gamma=0.5)
    data_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    train(model_coarse, optim, schedular, data_loader, device=device, near=2, far=6, epochs=16, model_fine=model_fine, N_samples=192, H=400, W=400)