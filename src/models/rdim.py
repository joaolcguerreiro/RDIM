import os
import sys
import math
import torch
import torch.nn as nn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.utils import upsample


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, num_embedding_dims):
        super().__init__()

        self.num_embedding_dims = num_embedding_dims

        # Get indexes from even embedding dimensions
        even_indexes = torch.arange(0, self.num_embedding_dims, 2)

        # Compute division term for positional embedding
        log_term = torch.tensor(1e4).log() / self.num_embedding_dims
        div_term = torch.exp(even_indexes * -log_term)

        # Register div_term as a buffer
        self.register_buffer('div_term', div_term)

    def forward(self, position):
        # Compute positional embedding
        position = position.unsqueeze(-1)
        sin_emb = torch.sin(position * self.div_term)
        cos_emb = torch.cos(position * self.div_term)

        # Interleave sine and cosine embeddings
        embedding = torch.stack([sin_emb, cos_emb], dim=-1).view(-1, 2 * ((self.num_embedding_dims + 1) // 2))

        # Slice out last dimension when the number of dimensions is uneven
        embedding = embedding[..., :self.num_embedding_dims]

        return embedding


class ConvolutionalBlock(nn.Module):
    def __init__(self, num_spatial_dimensions, in_channels, out_channels, num_groups=16, activation=True, max_pool=False, conv_first=True, **kwargs):
        super().__init__()

        # Convolutional Layer
        self.conv = (nn.Conv3d if num_spatial_dimensions == 3 else nn.Conv2d)(in_channels, out_channels, **kwargs)

        # Group Normalization Layer
        self.gn = nn.GroupNorm(num_groups, out_channels if conv_first else in_channels) if num_groups > 0 else nn.Identity()

        # Activation Function
        self.act = nn.SiLU() if activation else nn.Identity()

        # Max Pooling Layer
        self.pool = (nn.MaxPool3d if num_spatial_dimensions == 3 else nn.MaxPool2d)(kernel_size=2) if max_pool else nn.Identity()

        # Set Convolutional Layer Order
        self.conv_first = conv_first

    def forward(self, x):
        return self.pool(self.act(self.gn(self.conv(x)))) if self.conv_first else self.pool(self.conv(self.act(self.gn(x))))


class UpsampleBlock(nn.Module):
    def __init__(self, num_spatial_dimensions, in_channels, out_channels, scale_factor=2, mode='nearest', activation=False, **kwargs):
        super().__init__()

        # Upsample Layer
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=(False if mode != 'nearest' else None))

        # Convolutional Layer
        self.conv = (nn.Conv3d if num_spatial_dimensions == 3 else nn.Conv2d)(in_channels, out_channels, **kwargs)

        # Activation Function
        self.act = nn.SiLU() if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(self.upsample(x)))


class LayerNormChannel(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)

        num_spatial_dims = x.dim() - 2
        weight = self.weight.view(1, -1, *([1] * num_spatial_dims))
        bias = self.bias.view(1, -1, *([1] * num_spatial_dims))

        return x * weight + bias


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)

        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    def __init__(self, num_spatial_dimensions, c):
        super().__init__()

        self.pool = (nn.AdaptiveAvgPool3d if num_spatial_dimensions == 3 else nn.AdaptiveAvgPool2d)(1)
        self.conv = (nn.Conv3d if num_spatial_dimensions == 3 else nn.Conv2d)(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        return x * self.conv(self.pool(x))


class NAFBlock(nn.Module):
    def __init__(self, num_spatial_dimensions, c, num_time_embedding_dims, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        conv_func = nn.Conv3d if num_spatial_dimensions == 3 else nn.Conv2d

        self.norm1 = LayerNormChannel(c)
        self.conv1 = conv_func(c, dw_channel, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv2 = conv_func(dw_channel, dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)

        # Time embedding projection layer
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(num_time_embedding_dims, dw_channel)
        )

        self.sg = SimpleGate()
        self.sca = SimplifiedChannelAttention(num_spatial_dimensions, dw_channel // 2)
        self.conv3 = conv_func(dw_channel // 2, c, kernel_size=1, padding=0, stride=1, bias=True)

        self.norm2 = LayerNormChannel(c)
        ffn_channel = FFN_Expand * c
        self.conv4 = conv_func(c, ffn_channel, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv5 = conv_func(ffn_channel // 2, c, kernel_size=1, padding=0, stride=1, bias=True)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # NAFNet learns weighting parameters for skip connections
        self.beta = nn.Parameter(torch.zeros((1, c, *([1] * num_spatial_dimensions))), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, *([1] * num_spatial_dimensions))), requires_grad=True)

    def forward(self, inp, t_cond):
        x = inp
        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)

        # Apply time embedding projection
        t_cond_proj = self.time_proj(t_cond)
        num_diff_dims = x.dim() - t_cond_proj.dim()
        t_cond_proj = t_cond_proj.view(*t_cond_proj.shape, *([1] * num_diff_dims))

        x = x + t_cond_proj

        x = self.sg(x)
        x = self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNetPredictor(nn.Module):
    def __init__(self, num_spatial_dimensions=3, in_channels=1, out_channels=1, num_cond_features=64, num_inner_features=16, enc_blk_nums=(1, 1, 1, 28), middle_blk_num=1, dec_blk_nums=(1, 1, 1, 1), drop_out_rate=0., scale_factor=1):
        super().__init__()
        
        # Number of Spatial Dimensions
        self.num_spatial_dimensions = num_spatial_dimensions

        # Positional Encoding & MLP for time
        self.time_positional_encoding = SinusoidalPositionalEncoding(num_inner_features)
        self.time_mlp = nn.Sequential(
            nn.Linear(num_inner_features, num_inner_features * 4),
            nn.SiLU(),
            nn.Linear(num_inner_features * 4, num_inner_features)
        )

        # Upscale block for condition
        self.upsample_cond_block = UpsampleBlock(
            num_spatial_dimensions,
            num_cond_features,
            num_inner_features,
            scale_factor=scale_factor,
            mode='trilinear' if num_spatial_dimensions == 3 else 'bilinear',
            activation=False,
            kernel_size=3, stride=1, padding=1
        ) if torch.any(torch.tensor(scale_factor) > 1) or num_cond_features != num_inner_features else nn.Identity()

        # Intro
        conv_func = nn.Conv3d if num_spatial_dimensions == 3 else nn.Conv2d
        self.intro = conv_func(
            in_channels=in_channels, out_channels=num_inner_features, kernel_size=3, padding=1, stride=1, bias=True
        )

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()

        chan = num_inner_features
        for num in enc_blk_nums:
            self.encoders.append(
                nn.ModuleList([
                    NAFBlock(num_spatial_dimensions, chan, num_inner_features, drop_out_rate=drop_out_rate) for _ in range(num)
                ])
            )
            self.downs.append(
                conv_func(chan, 2 * chan, kernel_size=2, stride=2)
            )
            chan = chan * 2

        self.middle_blks = nn.ModuleList([
            NAFBlock(num_spatial_dimensions, chan, num_inner_features, drop_out_rate=drop_out_rate) for _ in range(middle_blk_num)
        ])

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    conv_func(chan, chan // 2, kernel_size=1, bias=False),
                    nn.Upsample(scale_factor=2, mode='nearest')
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.ModuleList([
                    NAFBlock(num_spatial_dimensions, chan, num_inner_features, drop_out_rate=drop_out_rate) for _ in range(num)
                ])
            )

        self.ending = conv_func(
            in_channels=num_inner_features, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=True
        )
        self.padder_size = 2 ** len(self.encoders)

    def initialize_weights(self, scale=1.0):
        for module in self.modules():
            if isinstance(module, (nn.Conv3d if self.num_spatial_dimensions == 3 else nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight.data)
                module.weight.data *= scale
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0)
            elif isinstance(module, (nn.GroupNorm, LayerNormChannel)):
                nn.init.constant_(module.weight.data, 1)
                nn.init.constant_(module.bias.data, 0)

    def forward(self, x, x_cond, timestep):
        assert torch.all((torch.tensor(x.shape) % torch.tensor(x_cond.shape))[-self.num_spatial_dimensions:] == 0), f'Input spatial shape ({tuple(x.shape[-self.num_spatial_dimensions:])}) must be divisible by conditioning variable spatial shape ({tuple(x_cond.shape[-self.num_spatial_dimensions:])}) in every dimension'

        pad_multiple = self.padder_size
        scale_factor = (torch.tensor(x.shape) // torch.tensor(x_cond.shape))[-self.num_spatial_dimensions:]

        if not torch.all((pad_multiple % scale_factor == 0) | (scale_factor % pad_multiple == 0)):
            pad_multiple = torch.where((pad_multiple % scale_factor == 0) | (scale_factor % pad_multiple == 0), pad_multiple, pad_multiple * scale_factor)

        x_pad = (pad_multiple - (torch.tensor(x.shape[-self.num_spatial_dimensions:]) % pad_multiple)) % pad_multiple
        x_cond_pad = x_pad // scale_factor

        x = long_range_skip_connection = nn.functional.pad(x, pad=list(reversed(torch.stack((x_pad, torch.zeros_like(x_pad)), dim=1).flatten())), mode='reflect')
        x_cond = nn.functional.pad(x_cond, pad=list(reversed(torch.stack((x_cond_pad, torch.zeros_like(x_cond_pad)), dim=1).flatten())), mode='reflect')

        t_cond = self.time_positional_encoding(timestep)
        t_cond = self.time_mlp(t_cond)

        x_cond_up = self.upsample_cond_block(x_cond)

        out = self.intro(x)
        out = out + x_cond_up

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            for blk in encoder:
                out = blk(out, t_cond)
            encs.append(out)
            out = down(out)

        for blk in self.middle_blks:
            out = blk(out, t_cond)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, reversed(encs)):
            out = up(out)
            out = out + enc_skip
            for blk in decoder:
                out = blk(out, t_cond)

        out = self.ending(out)
        out = out + long_range_skip_connection

        slicer = [slice(None)] * (out.dim() - self.num_spatial_dimensions) + [slice(None, -pad if pad > 0 else None) for pad in x_pad]
        out = out[tuple(slicer)]

        return out


class DenseBlock(nn.Module):
    def __init__(self, num_spatial_dimensions, in_channels, num_growth_features=32, num_conv_blocks=3, **kwargs):
        super().__init__()

        # Convolutional Blocks
        self.conv_blocks = nn.ModuleList()

        for block_index in range(num_conv_blocks):
            self.conv_blocks.append(
                ConvolutionalBlock(
                    num_spatial_dimensions,
                    in_channels + block_index * num_growth_features,
                    num_growth_features if block_index != (num_conv_blocks - 1) else in_channels,
                    num_groups=0,
                    activation=(True if block_index != (num_conv_blocks - 1) else False),
                    max_pool=False,
                    **kwargs
                )
            )

    def forward(self, x):
        out = skip_connections = x

        for conv_block in self.conv_blocks:
            out = conv_block(skip_connections)

            skip_connections = torch.cat([skip_connections, out], dim=1)

        return out


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, num_spatial_dimensions, in_channels, num_db_growth_features=32, num_db_blocks=2, num_db_conv_blocks=3, beta=0.2, **kwargs):
        super().__init__()

        # Residual Scale Factor
        self.beta = beta

        # Dense Blocks
        self.dense_blocks = nn.ModuleList()

        for _ in range(num_db_blocks):
            self.dense_blocks.append(
                DenseBlock(
                    num_spatial_dimensions,
                    in_channels,
                    num_growth_features=num_db_growth_features,
                    num_conv_blocks=num_db_conv_blocks,
                    **kwargs
                )
            )

    def forward(self, x):
        out = skip_connection = x

        for dense_block in self.dense_blocks:
            dense_out = dense_block(out)

            out = out + dense_out * self.beta

        out = out * self.beta + skip_connection

        return out


class LREncoder(nn.Module):
    def __init__(self, num_spatial_dimensions=3, in_channels=1, out_channels=1, num_features=64, num_rrdb_blocks=1, num_rrdb_db_blocks=2, num_rrdb_db_conv_blocks=3, num_rrdb_db_growth_features=32, scale_factor=1):
        super().__init__()

        # Number of Spatial Dimensions
        self.num_spatial_dimensions = num_spatial_dimensions

        # Input Layer
        self.input = ConvolutionalBlock(
            num_spatial_dimensions,
            in_channels,
            num_features,
            num_groups=0,
            activation=False,
            max_pool=False,
            kernel_size=3, stride=1, padding=1
        )

        # Residual in Residual Dense Blocks
        self.res_in_res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(
            num_spatial_dimensions,
            num_features,
            num_db_growth_features=num_rrdb_db_growth_features,
            num_db_blocks=num_rrdb_db_blocks,
            num_db_conv_blocks=num_rrdb_db_conv_blocks,
            kernel_size=3, stride=1, padding=1
        ) for _ in range(num_rrdb_blocks)])

        # Convolutional Layer
        self.conv = ConvolutionalBlock(
            num_spatial_dimensions,
            num_features,
            num_features,
            num_groups=0,
            activation=False,
            max_pool=False,
            kernel_size=3, stride=1, padding=1
        )

        # Upsampling Blocks
        self.upsample_blocks = nn.Sequential(*[UpsampleBlock(
            num_spatial_dimensions,
            num_features,
            num_features,
            scale_factor=2,
            mode='trilinear' if num_spatial_dimensions == 3 else 'bilinear',
            activation=True,
            kernel_size=3, stride=1, padding=1
        ) for _ in range(int(math.log2(scale_factor)))])

        # Output Block
        self.output = nn.Sequential(
            ConvolutionalBlock(
                num_spatial_dimensions,
                num_features,
                num_features,
                num_groups=0,
                activation=True,
                max_pool=False,
                kernel_size=3, stride=1, padding=1
            ),
            ConvolutionalBlock(
                num_spatial_dimensions,
                num_features,
                out_channels,
                num_groups=0,
                activation=False,
                max_pool=False,
                kernel_size=3, stride=1, padding=1
            )
        )

    def initialize_weights(self, scale=1.0):
        for module in self.modules():
            if isinstance(module, (nn.Conv3d if self.num_spatial_dimensions == 3 else nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight.data)
                module.weight.data *= scale
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0)

    def forward(self, x):
        out = skip_connection = self.input(x)

        out = self.res_in_res_blocks(out)

        out = self.conv(out) + skip_connection

        return out

    def pre_train_forward(self, x):
        out = self.forward(x)

        out = self.upsample_blocks(out)

        out = self.output(out)

        return out


class GaussianDiffusion(nn.Module):
    def __init__(self, num_timesteps=100, num_sampling_timesteps=1, sampling_timestep_schedule='LINEAR', beta_schedule='CIRCULAR', gamma=0.0, eta=0.0, num_spatial_dimensions=3, scale_factor=1, in_channels=1, out_channels=1, lr_condition_encoding=False, **kwargs):
        super().__init__()

        self.num_timesteps = num_timesteps
        self.gamma = gamma
        self.eta = eta
        self.num_spatial_dimensions = num_spatial_dimensions
        self.scale_factor = scale_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lr_condition_encoding = lr_condition_encoding

        # Register buffers
        sampling_timesteps = self.compute_reverse_trajectory_timesteps(sampling_timestep_schedule=sampling_timestep_schedule, num_sampling_timesteps=num_sampling_timesteps, num_timesteps=num_timesteps)
        betas, lambdas = self.compute_beta_schedule(beta_schedule=beta_schedule, num_timesteps=num_timesteps, **kwargs)

        self.register_buffer('sampling_timesteps', sampling_timesteps)
        self.register_buffer('betas', betas)
        self.register_buffer('lambdas', lambdas)

    @staticmethod
    def general_sigmoid_beta_schedule(num_timesteps=100, p=2):
        """
        General sigmoid-like schedule.
        """

        steps = torch.linspace(start=0, end=num_timesteps, steps=(num_timesteps + 1))
        general_sigmoid_function = torch.zeros_like(steps)
        general_sigmoid_function[1:] = 1 / (1 + ((num_timesteps / steps[1:]) - 1) ** p)
        lambdas = general_sigmoid_function[1:] - general_sigmoid_function[:-1]
        betas = general_sigmoid_function[1:]

        return betas, lambdas

    @staticmethod
    def general_sigmoid_antiderivative_beta_schedule(num_timesteps=100, p=10):
        """
        General sigmoid-like antiderivative schedule.
        """

        steps = torch.linspace(start=0, end=num_timesteps, steps=(num_timesteps + 1))
        general_sigmoid_antiderivative_function = (torch.log(1 + torch.exp(p * ((steps / num_timesteps) - (1 / 2)))) / math.log(1 + math.exp(p / 2))) * ((steps / num_timesteps) ** (1 - torch.sign(torch.abs(torch.tensor(p)))))
        div_function = (general_sigmoid_antiderivative_function - general_sigmoid_antiderivative_function[0]) / (general_sigmoid_antiderivative_function[-1] - general_sigmoid_antiderivative_function[0])
        lambdas = div_function[1:] - div_function[:-1]
        betas = div_function[1:]

        return betas, lambdas

    @staticmethod
    def circular_beta_schedule(num_timesteps=100, p=5):
        """
        Circular schedule.
        """

        steps = torch.linspace(start=0, end=num_timesteps, steps=(num_timesteps + 1))
        circular_function = (steps / (num_timesteps + (p - 1) * (num_timesteps - steps)))
        lambdas = circular_function[1:] - circular_function[:-1]
        betas = circular_function[1:]

        return betas, lambdas

    @staticmethod
    def parabolic_beta_schedule(num_timesteps=100, p=2):
        """
        Parabolic schedule.
        """

        steps = torch.linspace(start=0, end=num_timesteps, steps=(num_timesteps + 1))
        parabolic_function = (steps / num_timesteps) ** p
        lambdas = parabolic_function[1:] - parabolic_function[:-1]
        betas = parabolic_function[1:]

        return betas, lambdas

    def compute_beta_schedule(self, beta_schedule='CIRCULAR', num_timesteps=100, **kwargs):
        if beta_schedule == 'GENERAL_SIGMOID':
            return self.general_sigmoid_beta_schedule(num_timesteps=num_timesteps, **kwargs)
        elif beta_schedule == 'GENERAL_SIGMOID_ANTIDERIVATIVE':
            return self.general_sigmoid_antiderivative_beta_schedule(num_timesteps=num_timesteps, **kwargs)
        elif beta_schedule == 'CIRCULAR':
            return self.circular_beta_schedule(num_timesteps=num_timesteps, **kwargs)
        elif beta_schedule == 'PARABOLIC':
            return self.parabolic_beta_schedule(num_timesteps=num_timesteps, **kwargs)
        else:
            raise NotImplementedError(f'Beta schedule {beta_schedule} not implemented.')

    @staticmethod
    def compute_reverse_trajectory_timesteps(sampling_timestep_schedule='LINEAR', num_sampling_timesteps=1, num_timesteps=100):
        assert 1 <= num_sampling_timesteps <= num_timesteps, f'Number of sampling timesteps ({num_sampling_timesteps}) must be (a) greater than or equal to 1 and (b) less than or equal to ({num_timesteps})'

        if sampling_timestep_schedule == 'LINEAR':
            return reversed(torch.linspace(0, num_timesteps, steps=num_sampling_timesteps + 1, dtype=torch.long))[:-1]
        elif sampling_timestep_schedule == 'GEOMETRIC':
            geometric_steps = torch.exp(torch.linspace(0, math.log(num_timesteps), steps=num_sampling_timesteps + 1)).floor().long()[1:]

            # Compute unique geometric steps
            steps = []
            prev_step = 0

            for step in geometric_steps:
                # Ensure step > prev_step
                step = max(step, prev_step + 1)

                # Ensure step <= num_timesteps
                step = min(step, num_timesteps)

                # Save step and update previous step
                steps.append(step)
                prev_step = step

            # Ensure the last step is exactly num_timesteps
            steps[-1] = num_timesteps

            return reversed(torch.tensor(steps, dtype=torch.long))
        else:
            raise NotImplementedError(f'Sampling timestep schedule {sampling_timestep_schedule} not implemented.')

    def sample_timesteps(self, num_samples):
        return torch.randint(low=1, high=self.num_timesteps + 1, size=(num_samples,), dtype=torch.long)

    def q_sample(self, x_0, delta, timestep):
        """
        Compute the volume at a specific timestep (t) given the initial volume and residual delta. q(x_t|x_0, delta).
        """

        assert torch.all((timestep >= 1) & (timestep <= self.num_timesteps)), f'Timestep(s) ({timestep}) must be (a) greater than or equal to 1 and (b) less than or equal to ({self.num_timesteps})'

        # Get timestep index
        timestep_index = timestep - 1

        # Count dimension mismatch between volume and timestep
        num_diff_dims = x_0.dim() - timestep.dim()

        # Compute beta at given timestep and match volume dimensions
        beta_t = self.betas[timestep_index].view(*timestep.shape, *([1] * num_diff_dims))
        gamma = torch.tensor(self.gamma, device=timestep.device)

        # Sample random noise
        eps = torch.randn_like(x_0)

        x_t = x_0 - beta_t * delta + torch.sqrt(torch.pow(gamma, 2) * beta_t) * eps

        return x_t

    def p_sample(self, x_tau_k, x_0, y_0, tau_k, tau_k_m1):
        """
        Compute the volume at the distant timestep (tau_k - 1) given the volume at a specific timestep (tau_k) and the LQ data. p_θ(x_tau_k_m1|x_tau_k, y_0).
        """

        assert torch.all((tau_k >= 1) & (tau_k <= self.num_timesteps)), f'Timestep(s) ({tau_k}) must be (a) greater than or equal to 1 and (b) less than or equal to ({self.num_timesteps})'
        assert torch.all((tau_k_m1 >= 0) & (tau_k_m1 < self.num_timesteps)), f'Previous timestep(s) ({tau_k_m1}) must be (a) greater than or equal to 0 and (b) less than ({self.num_timesteps})'

        # Define condition to check if timestep is greater than last sampling timestep (tau_1)
        timestep_condition = tau_k > int(self.sampling_timesteps[-1])

        # Define condition to check if gamma is not 0
        gamma_condition = torch.tensor(self.gamma != 0, device=tau_k.device)

        # Get timestep index
        tau_k_index = tau_k - 1
        tau_k_m1_index = tau_k_m1 - 1

        # Count dimension mismatch between volume and timestep
        num_diff_dims = x_tau_k.dim() - tau_k.dim()

        # Compute diffusion parameters and match volume dimensions
        beta_tau_k = self.betas[tau_k_index].view(*tau_k.shape, *([1] * num_diff_dims))
        beta_tau_k_m1 = torch.where(timestep_condition, self.betas[tau_k_m1_index], 0).view(*tau_k.shape, *([1] * num_diff_dims))
        lambda_tau_k = self.lambdas[tau_k_index].view(*tau_k.shape, *([1] * num_diff_dims))
        gamma = torch.tensor(self.gamma, device=tau_k.device)
        eta = torch.tensor(self.eta, device=tau_k.device)

        # Sample random noise and match volume dimensions
        z = torch.randn_like(x_tau_k)

        # Estimate delta
        delta = x_0 - y_0

        # Compute distant volume at timestep tau_{k - 1}
        variance = torch.pow(eta, 2) * torch.pow(gamma, 2) * (beta_tau_k_m1 / beta_tau_k) * lambda_tau_k
        mean = torch.where(gamma_condition.expand(*tau_k.shape, *([1] * num_diff_dims)), x_0 - beta_tau_k_m1 * delta + torch.sqrt(torch.pow(gamma, 2) * beta_tau_k_m1 - variance) * ((x_tau_k - x_0 + beta_tau_k * delta) / torch.sqrt(torch.pow(gamma, 2) * beta_tau_k)), x_0 - beta_tau_k_m1 * delta)

        x_tau_k_m1 = mean + torch.sqrt(variance) * z

        return x_tau_k_m1

    def forward_diffusion(self, x_0, delta, timestep):
        assert torch.all((timestep >= 1) & (timestep <= self.num_timesteps)), f'Timestep(s) ({timestep}) must be (a) greater than or equal to 1 and (b) less than or equal to ({self.num_timesteps})'

        # Compute volume at a specific timestep
        x_t = self.q_sample(x_0, delta, timestep)

        return x_t

    def reverse_diffusion(self, x_t, x_0, y_0, timestep, timestep_m1):
        assert torch.all((timestep >= 1) & (timestep <= self.num_timesteps)), f'Timestep(s) ({timestep}) must be (a) greater than or equal to 1 and (b) less than or equal to ({self.num_timesteps})'
        assert torch.all((timestep_m1 >= 0) & (timestep_m1 < self.num_timesteps)), f'Previous timestep(s) ({timestep_m1}) must be (a) greater than or equal to 0 and (b) less than ({self.num_timesteps})'

        # Predict volume at the previous timestep
        x_t_m1_pred = self.p_sample(x_t, x_0, y_0, timestep, timestep_m1)

        return x_t_m1_pred

    @torch.no_grad()
    def sample(self, x_lr, noise_predictor, lr_encoder=None, noise_predictor_inferer=None):
        assert not self.lr_condition_encoding or lr_encoder, f'LR Encoder is required because LR condition encoding is {self.lr_condition_encoding}'

        # Set the noise predictor to evaluation mode
        noise_predictor.eval()

        # Set the LR encoder to evaluation mode if applicable
        if self.lr_condition_encoding:
            lr_encoder.eval()

        # Upsample volume
        x_lr_up = torch.stack([upsample(volume, self.scale_factor, self.num_spatial_dimensions) for volume in x_lr])

        # Get number of samples (batch size) and spatial dimensions
        num_samples, _, *spatial_shape = x_lr_up.shape

        # Sample last timestep
        x_t = x_lr_up + math.sqrt(math.pow(self.gamma, 2)) * torch.randn(num_samples, self.out_channels, *spatial_shape, device=x_lr_up.device)

        # Define previous sampling timesteps
        sampling_timesteps_m1 = torch.cat([self.sampling_timesteps[1:], torch.tensor([0], device=self.sampling_timesteps.device)], dim=0)

        for timestep, timestep_m1 in zip(self.sampling_timesteps, sampling_timesteps_m1):
            # Adapt timestep for batch processing
            timestep = torch.full((num_samples,), timestep, dtype=torch.long, device=timestep.device)
            timestep_m1 = torch.full((num_samples,), timestep_m1, dtype=torch.long, device=timestep_m1.device)

            # Predict initial volume
            x_0_pred = noise_predictor_inferer(inputs=x_t, conditions=x_lr, timesteps=timestep, inputs_network=noise_predictor, conditions_network=(lr_encoder if self.lr_condition_encoding else None), cache_conditions=True) if noise_predictor_inferer else noise_predictor(x_t, (lr_encoder(x_lr) if self.lr_condition_encoding else x_lr), timestep)

            # Reverse diffusion to compute volume at previous timestep
            x_t = self.reverse_diffusion(x_t, x_0_pred, x_lr_up, timestep, timestep_m1)

        # Clear cached memory
        if noise_predictor_inferer:
            noise_predictor_inferer.clear_conditions_cache()

        return x_t


if __name__ == '__main__':
    pass
