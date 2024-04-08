import math
from random import random

import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from util import *


class GaussianDiffusion(pl.LightningModule):
    def __init__(self,
                 model,
                 image_size,
                 noise_schedule,
                 schedule_fn_kwargs,
                 channels=3,
                 self_cond=None,
                 pred_obj='v',
                 beta_schedule = 'sigmoid',
                 timesteps=1000,
                 num_sample_steps=500,
                 ddim_sampling_eta=0.):
        super().__init__()
        assert pred_obj in {'v', 'eps'}

        self.model = model

        self.channels = channels
        self.image_size = image_size
        self.num_timesepts = timesteps
        self.pred_obj = pred_obj

        # noise schedule 관련 정의하기
        if beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError("check the beta schedule")

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1,0), value=1.)

        # sampling with ddim
        assert num_sample_steps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # function to register buffer -> float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumporod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # diffusion q(x_t | x_{t-1})
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))  # to avoid high variance, set min while clamping
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))


        # signal noise ratio
        snr = alphas_cumprod / (1. - alphas_cumprod)
        maybe_clipped_snr = snr.clone()

        if pred_obj == 'v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1)) # sigmoid
        elif pred_obj == 'eps':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        self.normalize = normalize_to_neg_one_to_one # normalization of data [0, 1] -> [-1, 1]
        self.unnormalize = unnormalize_to_zero_to_one # unnormalization of data [-1, 1] -> [0, 1]

        @property
        def device(self):
            return self.betas.device

        def predict_start_from_noise(self, x_t, t, noise):
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
            )

        def predict_v(self, x_start, t, noise):
            return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
            )

        def predict_start_from_v(self, x_t, t, v):
            return (
                extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
            )

        def q_posterior(self, x_start, x_t, t):
            posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
            )
            posterior_variance = extract(self.posterior_variance, t, x_t.shape)
            posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
            return posterior_mean, posterior_variance, posterior_log_variance_clipped


        def p_mean_variance(self, x, t, x_self_cond, clip_denoised = True):
            preds = self.model_predictions(x, t, x_self_cond)
            x_start = preds.pred_x_start

            if clip_denoised:
                x_start.clamp_(-1., 1.)

            model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x_t, t = t)

            return model_mean, posterior_variance, posterior_log_variance, x_start

        @torch.no_grad()
        def p_sample(self, x, t, x_self_cond, clip_denoised):
            b, *_ = x.shape
            batched_times = torch.full((b,), t, device = self.device, dtype = torch.long) # b x timesteps
            model_mean, _, model_log_var, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True)
            noise = torch.randn_like(x) if t > 0 else 0. # t == 0 -> noise X
            x_pred = model_mean + (0.5 * model_log_var).exp() * noise
            return x_pred, x_start

        @torch.no_grad()
        def p_sample_loop(self, shape, return_all_timesteps=False):
            batch = shape[0]

            img = torch.randn(shape, device = self.device)
            imgs = [img]

            x_start = None

            for t in tqdm(reversed(range(0, self.num_timesteps)), desc = "sampling loop"):
                self_cond = x_start if self.self_cond else None
                img, x_start = self.p_sample(img, t, self_cond)
                imgs.append(img)
            if return_all_timesteps:
                return imgs
            return img

        @torch.no_grad()
        def sample(self, batch_size=16, return_all_timesteps=False):
            image_size = self.image_size
            channels = self.channels
            return self.p_sample_loop((batch_size, channels, image_size, image_size), return_all_timesteps=return_all_timesteps)


        def model_predictions(self, x, t, x_self_cond=None):
            model_output = self.model(x, t, x_self_cond)

            if self.pred_obj == 'eps':
                pred_noise = model_output
                x_start = self.predict_start_from_noise(x, t, pred_noise)
            elif self.pred_obj == 'v':
                v = model_output
                x_start = self.predict_start_from_v(x, t, v)
                pred_noise = self.predict_start_from_noise(x, t, x_start)

            return pred_noise, x_start

        def p_losses(self, x_start, t):
            b, c, h, w = x_start.shape

            noise = torch.randn_like(x_start)

            x_noisy = self.q_sample(x_start = x_start, t = t, noise = noise)

            # self-conditioning with unet -> slower training by 25% / lower FID !
            x_self_cond = None
            if exists(self.self_cond) and random() < 0.5:
                with torch.no_grad():
                    _, x_self_cond = self.model_predictions(x_noisy, t) # pred_x_start
                    x_self_cond.detach_()

            # prediction, gradient step
            model_out = self.model(x_noisy, t, x_self_cond)

            if self.pred_obj == 'eps':
                target = noise
            if self.pred_obj == 'v':
                target = self.predict_v(x_start, t, noise)

            # loss
            loss = F.mse_loss(model_out, target, reduction='none')
            loss = reduce(loss, 'b ... -> b', 'mean')

            loss = loss * extract(self.loss_weight, t, loss.shape)
            return loss.mean()

        def forward(self, img, *args, **kwargs):
            b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
            assert h == img_size and w == img_size
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

            img = self.normalize(img)
            return self.p_losses(img, t, *args, **kwargs)

        def training_step(self, batch, batch_idx):
            x = batch
            if len(x.shape) == 3:
                x = x[..., None] # 13D data
            x = rearrange(x, 'b h w c -> b c h w').float()

            loss = self(x)

            # self.log or self.log_dict

            # if self.use_schedular:
            #     lr = self.optimizers().param_groups[0]['lr']
            #     # self.log

            return loss

        def configure_optimizers(self):
            lr = self.learning_rate
            params = list(self.model.parameters())
            opt = torch.optim.AdamW(params, lr=lr)
            return opt
