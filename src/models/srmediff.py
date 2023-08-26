import torch
import torch.nn as nn


class NoisePredictor:
    pass


class Diffusion(nn.Module):
    def __init__(self, num_steps=1000, beta_schedule='COSINE', residual_prediction=True, scale_factor=2, **kwargs):
        super().__init__()

        self.num_steps = num_steps
        self.betas = self.compute_betas_schedule(beta_schedule, **kwargs)
        self.alphas = 1.0 - self.betas
        self.alphas_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.residual_prediction = residual_prediction
        self.scale_factor = scale_factor

    def compute_betas_schedule(self, beta_schedule='COSINE', beta_offset=8e-3, beta_max=9.99e-1, beta_start=1e-4, beta_end=2e-2):
        if beta_schedule == 'LINEAR':
            return torch.linspace(start=beta_start, end=beta_end, steps=self.num_steps)
        elif beta_schedule == 'COSINE':
            num_steps = self.num_steps + 1
            steps = torch.linspace(start=0, end=num_steps, steps=num_steps)
            cosine_function = torch.cos((steps / num_steps + beta_offset) / (1 + beta_offset) * (torch.pi / 2)) ** 2
            alphas_cum_prod = cosine_function / cosine_function[0]
            betas = 1 - (alphas_cum_prod[1:] / alphas_cum_prod[:-1])

            return torch.clamp(betas, min=0.0, max=beta_max)
        else:
            raise NotImplementedError(f'Beta schedule {beta_schedule} not implemented.')

    def compute_volume_at_timestep(self, x_0, timestep):
        # Get number of volume dimensions
        num_dims = x_0.dim()

        # Reshape to match volume dimensions
        alpha_cum_prod = self.alphas_cum_prod[timestep].view(*([1] * num_dims))

        # Compute random noise
        eps = torch.rand_like(x_0)

        return torch.sqrt(alpha_cum_prod) * x_0 + torch.sqrt(1 - alpha_cum_prod) * eps

    def compute_volume_at_next_timestep(self, x_t):
        pass















