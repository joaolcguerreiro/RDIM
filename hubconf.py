import torch
from src.utils.inferer import MultiScaleConditionalSlidingWindowInferer
from src.models.rdim import GaussianDiffusion, NoisePredictor, LREncoder

dependencies = ['torch']


def load_model(checkpoint_url=None, pretrained=False, gaussian_diffusion_kwargs=None, noise_predictor_kwargs=None, lr_encoder_kwargs=None):
    if checkpoint_url:
        # Load the pre-trained checkpoint
        checkpoint = torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu', weights_only=False)

        # Override checkpoint kwargs with input kwargs
        checkpoint_gaussian_diffusion_kwargs = {
            'num_timesteps': checkpoint['gaussian_diffusion_num_timesteps'],
            'num_sampling_timesteps': checkpoint['gaussian_diffusion_num_sampling_timesteps'],
            'sampling_timestep_schedule': checkpoint['gaussian_diffusion_sampling_timestep_schedule'],
            'beta_schedule': checkpoint['gaussian_diffusion_beta_schedule'],
            'gamma': checkpoint['gaussian_diffusion_gamma'],
            'eta': checkpoint['gaussian_diffusion_eta'],
            'num_spatial_dimensions': checkpoint['gaussian_diffusion_num_spatial_dimensions'],
            'scale_factor': checkpoint['gaussian_diffusion_scale_factor'],
            'in_channels': checkpoint['gaussian_diffusion_in_channels'],
            'out_channels': checkpoint['gaussian_diffusion_out_channels'],
            'lr_condition_encoding': checkpoint['gaussian_diffusion_lr_condition_encoding'],
            **{key.lower(): value for key, value in checkpoint['gaussian_diffusion_beta_schedule_parameters'].items()}
        }
        checkpoint_gaussian_diffusion_kwargs.update(gaussian_diffusion_kwargs or {})
        gaussian_diffusion_kwargs = checkpoint_gaussian_diffusion_kwargs

        checkpoint_noise_predictor_kwargs = {
            'num_spatial_dimensions': checkpoint['gaussian_diffusion_num_spatial_dimensions'],
            'in_channels': checkpoint['gaussian_diffusion_out_channels'],
            'out_channels': checkpoint['gaussian_diffusion_out_channels'],
            'num_cond_features': (checkpoint['lr_encoder_num_features'] if checkpoint['gaussian_diffusion_lr_condition_encoding'] else checkpoint['gaussian_diffusion_in_channels']),
            'num_inner_features': checkpoint['noise_predictor_num_inner_features'],
            'feature_multipliers': checkpoint['noise_predictor_feature_multipliers'],
            'num_rb_blocks': checkpoint['noise_predictor_num_rb_blocks'],
            'num_rb_conv_blocks': checkpoint['noise_predictor_num_rb_conv_blocks'],
            'attention_levels': checkpoint['noise_predictor_attention_levels'],
            'num_attention_heads': checkpoint['noise_predictor_num_attention_heads'],
            'num_groups': checkpoint['noise_predictor_num_groups'],
            'scale_factor': checkpoint['gaussian_diffusion_scale_factor'],
        }
        checkpoint_noise_predictor_kwargs.update(noise_predictor_kwargs or {})
        noise_predictor_kwargs = checkpoint_noise_predictor_kwargs

        if checkpoint['gaussian_diffusion_lr_condition_encoding']:
            checkpoint_lr_encoder_kwargs = {
                'num_spatial_dimensions': checkpoint['gaussian_diffusion_num_spatial_dimensions'],
                'in_channels': checkpoint['gaussian_diffusion_in_channels'],
                'out_channels': checkpoint['gaussian_diffusion_out_channels'],
                'num_features': checkpoint['lr_encoder_num_features'],
                'num_rrdb_blocks': checkpoint['lr_encoder_num_rrdb_blocks'],
                'num_rrdb_db_blocks': checkpoint['lr_encoder_num_rrdb_db_blocks'],
                'num_rrdb_db_conv_blocks': checkpoint['lr_encoder_num_rrdb_db_conv_blocks'],
                'num_rrdb_db_growth_features': checkpoint['lr_encoder_num_rrdb_db_growth_features'],
                'scale_factor': checkpoint['gaussian_diffusion_scale_factor'],
            }
            checkpoint_lr_encoder_kwargs.update(lr_encoder_kwargs or {})
            lr_encoder_kwargs = checkpoint_lr_encoder_kwargs

    # Initialize models
    gaussian_diffusion = GaussianDiffusion(**(gaussian_diffusion_kwargs or {}))
    noise_predictor = NoisePredictor(**(noise_predictor_kwargs or {}))
    lr_encoder = LREncoder(**(lr_encoder_kwargs or {})) if gaussian_diffusion.lr_condition_encoding else None

    # Initialize models weights
    if checkpoint_url and pretrained:
        noise_predictor.load_state_dict(checkpoint['noise_predictor_state_dict'])
        lr_encoder.load_state_dict(checkpoint['lr_encoder_state_dict']) if gaussian_diffusion.lr_condition_encoding else None

    return gaussian_diffusion, noise_predictor, lr_encoder


def load_inferer(**kwargs):
    # Define patch-based inferer
    inferer = MultiScaleConditionalSlidingWindowInferer(**kwargs)

    return inferer
