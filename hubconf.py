import torch
from src.utils.inferer import MultiScaleConditionalSlidingWindowInferer
from src.models.voxrdiff import GaussianDiffusion, NoisePredictor, LREncoder

dependencies = ['torch']


def load_model(checkpoint_url=None, pretrained=False, gaussian_diffusion_kwargs=None, noise_predictor_kwargs=None, lr_encoder_kwargs=None):
    # Initialize models
    gaussian_diffusion = GaussianDiffusion(**(gaussian_diffusion_kwargs or {}))
    noise_predictor = NoisePredictor(**(noise_predictor_kwargs or {}))
    lr_encoder = LREncoder(**(lr_encoder_kwargs or {})) if gaussian_diffusion.lr_condition_encoding else None

    if checkpoint_url and pretrained:
        # Load the pre-trained checkpoint
        checkpoint = torch.hub.load_state_dict_from_url(checkpoint_url, weights_only=False)

        # Initialize models' weights
        noise_predictor.load_state_dict(checkpoint['noise_predictor_state_dict'])
        lr_encoder.load_state_dict(checkpoint['lr_encoder_state_dict']) if gaussian_diffusion.lr_condition_encoding else None

    return gaussian_diffusion, noise_predictor, lr_encoder


def load_inferer(**kwargs):
    # Define patch-based inferer
    inferer = MultiScaleConditionalSlidingWindowInferer(**kwargs)

    return inferer
