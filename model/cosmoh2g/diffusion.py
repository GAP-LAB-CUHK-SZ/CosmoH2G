# Ancestral sampler adapted from https://github.com/openai/guided-diffusion
import math

import numpy as np
import torch

NUM_STEPS = 500


def cosine_betas(num_steps, max_beta=0.999):
    def alpha_bar(t):
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    return np.array(
        [min(1 - alpha_bar((i + 1) / num_steps) / alpha_bar(i / num_steps), max_beta) for i in range(num_steps)],
        dtype=np.float64,
    )


class Sampler:
    """x0-prediction DDPM sampler with the fixed small posterior variance."""

    def __init__(self, num_steps=NUM_STEPS):
        betas = cosine_betas(num_steps)
        alphas_cumprod = np.cumprod(1.0 - betas)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        self.num_steps = num_steps
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_log_variance = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.coef_x0 = betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.coef_xt = (1.0 - alphas_cumprod_prev) * np.sqrt(1.0 - betas) / (1.0 - alphas_cumprod)

    @staticmethod
    def _take(arr, t, shape):
        out = torch.from_numpy(arr).to(device=t.device)[t].float()
        while out.ndim < len(shape):
            out = out[..., None]
        return out.expand(shape)

    @torch.no_grad()
    def sample(self, model, shape, device, progress=False, **cond):
        x = torch.randn(*shape, device=device)
        steps = range(self.num_steps - 1, -1, -1)
        if progress:
            from tqdm.auto import tqdm

            steps = tqdm(steps)
        for i in steps:
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x0 = model(x, t, **cond)
            mean = self._take(self.coef_x0, t, x.shape) * x0 + self._take(self.coef_xt, t, x.shape) * x
            noise = torch.randn_like(x)
            nonzero = (t != 0).float().view(-1, *([1] * (x.ndim - 1)))
            x = mean + nonzero * torch.exp(0.5 * self._take(self.posterior_log_variance, t, x.shape)) * noise
        return x
