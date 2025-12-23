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

        # Max Polling Layer
        self.pool = (nn.MaxPool3d if num_spatial_dimensions == 3 else nn.MaxPool2d)(kernel_size=2) if max_pool else nn.Identity()

        # Set Convolutional Layer Order
        self.conv_first = conv_first

    def forward(self, x):
        return self.pool(self.act(self.gn(self.conv(x)))) if self.conv_first else self.pool(self.conv(self.act(self.gn(x))))


class DownsampleBlock(nn.Module):
    def __init__(self, num_spatial_dimensions, in_channels, out_channels, scale_factor=2, activation=False, **kwargs):
        super().__init__()

        # Convolutional Layer
        self.conv = (nn.Conv3d if num_spatial_dimensions == 3 else nn.Conv2d)(in_channels, out_channels, stride=scale_factor, padding_mode='reflect', **kwargs)

        # Activation Function
        self.act = nn.SiLU() if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))


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


class ResidualBlock(nn.Module):
    def __init__(self, num_spatial_dimensions, in_channels, out_channels, num_time_embedding_dims, num_conv_blocks=2, num_groups=16, **kwargs):
        super().__init__()
        assert num_conv_blocks > 0 if in_channels != out_channels else True, f'Number of convolutional blocks ({num_conv_blocks}) must be greater than 0'

        # Time embedding projection layer
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(num_time_embedding_dims, out_channels)
        )

        # Convolutional Blocks
        self.conv_blocks = nn.ModuleList()

        for block_index in range(num_conv_blocks):
            self.conv_blocks.append(
                ConvolutionalBlock(
                    num_spatial_dimensions,
                    out_channels if block_index != 0 else in_channels,
                    out_channels,
                    num_groups=num_groups,
                    activation=True,
                    max_pool=False,
                    conv_first=False,
                    **kwargs
                )
            )

        # Skip Connection Layer
        self.residual_conv = (nn.Conv3d if num_spatial_dimensions == 3 else nn.Conv2d)(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_cond):
        # Apply time embedding projection
        t_cond = self.time_proj(t_cond)

        # Count dimension mismatch between input tensor and timestep embedding
        num_diff_dims = x.dim() - t_cond.dim()

        # Reshape time embedding to match input tensor dimensions
        t_cond = t_cond.view(*t_cond.shape, *([1] * num_diff_dims))

        out = skip_connection = x

        for block_index, conv_block in enumerate(self.conv_blocks):
            out = conv_block(out)

            if block_index < len(self.conv_blocks) - 1 or block_index == 0:
                out = out + t_cond

        out = out + self.residual_conv(skip_connection)

        return out


class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, num_heads=16, num_groups=16):
        super().__init__()

        # Group Normalization Layer
        self.pre_gn = nn.GroupNorm(num_groups, in_channels) if num_groups > 0 else nn.Identity()

        # Multi-Head Attention Mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            batch_first=True
        )

        # Group Normalization Layer
        self.post_gn = nn.GroupNorm(num_groups, in_channels) if num_groups > 0 else nn.Identity()

    def forward(self, x):
        skip_connection = x

        # Apply normalization
        out = self.pre_gn(x)

        # Reshape and transpose normalized input tensor to process depth feature maps rather than spatial maps
        # [batch, in_channels, D, H, W] => [batch, D*H*W, in_channels] = [batch, sequence_len, num_embedding_dims]
        batch, in_channels, *spatial_shape = out.shape
        out = out.view(batch, in_channels, torch.tensor(spatial_shape).prod()).transpose(-2, -1).contiguous()

        # Apply Attention Mechanism
        out, _ = self.attention(query=out, key=out, value=out, need_weights=False)

        # Transpose and reshape output tensor to match input shape
        # [batch, D*H*W, in_channels] => [batch, in_channels, D, H, W]
        out = out.transpose(-2, -1).contiguous().view(batch, in_channels, *spatial_shape)

        # Add skip connection and apply normalization
        out = self.post_gn(out + skip_connection)

        return out


class ResidualAttentionBlock(nn.Module):
    def __init__(self, num_spatial_dimensions, in_channels, out_channels, num_time_embedding_dims, num_rb_conv_blocks=2, self_attention=True, num_attention_heads=16, num_groups=16, **kwargs):
        super().__init__()

        # Residual Block
        self.res_block = ResidualBlock(
            num_spatial_dimensions,
            in_channels,
            out_channels,
            num_time_embedding_dims=num_time_embedding_dims,
            num_conv_blocks=num_rb_conv_blocks,
            num_groups=num_groups,
            **kwargs
        )

        # Optional Self-Attention Block
        self.att_block = SelfAttentionBlock(
            out_channels,
            num_heads=num_attention_heads,
            num_groups=num_groups
        ) if self_attention else nn.Identity()

    def forward(self, x, t_cond):
        out = self.res_block(x, t_cond)

        out = self.att_block(out)

        return out


class EncoderBlock(nn.Module):
    def __init__(self, num_spatial_dimensions, in_channels, out_channels, num_time_embedding_dims, num_rb_blocks=2, num_rb_conv_blocks=2, self_attention=True, num_attention_heads=16, num_groups=16, **kwargs):
        super().__init__()

        # Residual Blocks with Optional Self-Attention Blocks
        self.res_att_blocks = nn.ModuleList()

        for block_index in range(num_rb_blocks):
            self.res_att_blocks.append(
                ResidualAttentionBlock(
                    num_spatial_dimensions,
                    out_channels if block_index != 0 else in_channels,
                    out_channels,
                    num_time_embedding_dims=num_time_embedding_dims,
                    num_rb_conv_blocks=num_rb_conv_blocks,
                    self_attention=self_attention,
                    num_attention_heads=num_attention_heads,
                    num_groups=num_groups,
                    **kwargs
                )
            )

        # Downsampling Block
        self.downsample_block = DownsampleBlock(
            num_spatial_dimensions,
            out_channels,
            out_channels,
            scale_factor=2,
            activation=False,
            kernel_size=kwargs.get('kernel_size', 3), padding=kwargs.get('padding', 1)
        )

    def forward(self, x, t_cond):
        out = x

        for res_att_block in self.res_att_blocks:
            out = res_att_block(out, t_cond)

        out = self.downsample_block(out)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, num_spatial_dimensions, in_channels, out_channels, num_time_embedding_dims, num_rb_blocks=2, num_rb_conv_blocks=2, self_attention=True, num_attention_heads=16, num_groups=16, **kwargs):
        super().__init__()

        # Residual Blocks with Optional Self-Attention Blocks
        self.res_att_blocks = nn.ModuleList()

        for block_index in range(num_rb_blocks):
            self.res_att_blocks.append(
                ResidualAttentionBlock(
                    num_spatial_dimensions,
                    out_channels if block_index != 0 else in_channels,
                    out_channels,
                    num_time_embedding_dims=num_time_embedding_dims,
                    num_rb_conv_blocks=num_rb_conv_blocks,
                    self_attention=self_attention,
                    num_attention_heads=num_attention_heads,
                    num_groups=num_groups,
                    **kwargs
                )
            )

        # Upsampling Block
        self.upsample_block = UpsampleBlock(
            num_spatial_dimensions,
            out_channels,
            out_channels,
            scale_factor=2,
            mode='trilinear' if num_spatial_dimensions == 3 else 'bilinear',
            activation=False,
            **kwargs
        )

    def forward(self, x, t_cond):
        out = x

        for res_att_block in self.res_att_blocks:
            out = res_att_block(out, t_cond)

        out = self.upsample_block(out)

        return out


class BottleneckBlock(nn.Module):
    def __init__(self, num_spatial_dimensions, in_channels, num_time_embedding_dims, num_rb_blocks=2, num_rb_conv_blocks=2, self_attention=True, num_attention_heads=16, num_groups=16, **kwargs):
        super().__init__()

        # Residual Blocks with Optional Self-Attention Blocks
        self.res_att_blocks = nn.ModuleList()

        for _ in range(num_rb_blocks):
            self.res_att_blocks.append(
                ResidualAttentionBlock(
                    num_spatial_dimensions,
                    in_channels,
                    in_channels,
                    num_time_embedding_dims=num_time_embedding_dims,
                    num_rb_conv_blocks=num_rb_conv_blocks,
                    self_attention=self_attention,
                    num_attention_heads=num_attention_heads,
                    num_groups=num_groups,
                    **kwargs
                )
            )

    def forward(self, x, t_cond):
        out = x

        for res_att_block in self.res_att_blocks:
            out = res_att_block(out, t_cond)

        return out


class NoisePredictor(nn.Module):
    def __init__(self, num_spatial_dimensions=3, in_channels=1, out_channels=1, num_cond_features=64, num_inner_features=128, feature_multipliers=(1, 2, 4, 8), num_rb_blocks=2, num_rb_conv_blocks=2, attention_levels=((False,) * 2 + (True,) * 2), num_attention_heads=16, num_groups=16, scale_factor=1):
        super().__init__()
        assert len(feature_multipliers) == len(attention_levels), f'Array size of attention levels ({len(attention_levels)}) must match array size of feature multipliers ({len(feature_multipliers)})'

        # Number of Spatial Dimensions
        self.num_spatial_dimensions = num_spatial_dimensions

        # Positional Encoding
        self.time_positional_encoding = SinusoidalPositionalEncoding(num_inner_features)

        # Multilayer Perceptron for time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(num_inner_features, num_inner_features * 4),
            nn.SiLU(),
            nn.Linear(num_inner_features * 4, num_inner_features)
        )

        # Upsample/Projection block for condition
        self.upsample_cond = UpsampleBlock(
            num_spatial_dimensions,
            num_cond_features,
            num_inner_features,
            scale_factor=scale_factor,
            mode='trilinear' if num_spatial_dimensions == 3 else 'bilinear',
            activation=False,
            kernel_size=3, stride=1, padding=1
        ) if scale_factor > 1 or num_cond_features != num_inner_features else nn.Identity()

        # Input Layer
        self.input = ConvolutionalBlock(
            num_spatial_dimensions,
            in_channels,
            num_inner_features,
            num_groups=0,
            activation=False,
            max_pool=False,
            kernel_size=3, stride=1, padding=1
        )

        # Encoder blocks for contracting/downsampling path
        self.encoder_blocks = nn.ModuleList()

        for block_index in range(len(feature_multipliers)):
            # Determine feature multipliers for current and previous layers
            multiplier = feature_multipliers[block_index]
            prev_multiplier = feature_multipliers[block_index - 1] if block_index != 0 else None

            self.encoder_blocks.append(
                EncoderBlock(
                    num_spatial_dimensions,
                    num_inner_features * prev_multiplier if prev_multiplier is not None else num_inner_features,
                    num_inner_features * multiplier,
                    num_time_embedding_dims=num_inner_features,
                    num_rb_blocks=num_rb_blocks,
                    num_rb_conv_blocks=num_rb_conv_blocks,
                    self_attention=attention_levels[block_index],
                    num_attention_heads=num_attention_heads,
                    num_groups=num_groups,
                    kernel_size=3, stride=1, padding=1
                )
            )

        # Bottleneck block
        self.bottleneck_block = BottleneckBlock(
            num_spatial_dimensions,
            num_inner_features * feature_multipliers[-1],
            num_time_embedding_dims=num_inner_features,
            num_rb_blocks=num_rb_blocks,
            num_rb_conv_blocks=num_rb_conv_blocks,
            self_attention=True,
            num_attention_heads=num_attention_heads,
            num_groups=num_groups,
            kernel_size=3, stride=1, padding=1
        )

        # Decoder blocks for expanding/upsampling path
        self.decoder_blocks = nn.ModuleList()

        # Get reversed feature multipliers and attention levels
        reversed_feature_multipliers = list(reversed(feature_multipliers))
        reversed_attention_levels = list(reversed(attention_levels))

        for block_index in range(len(reversed_feature_multipliers)):
            # Determine the feature multipliers for current and previous layers
            multiplier = reversed_feature_multipliers[block_index]
            prev_multiplier = reversed_feature_multipliers[block_index - 1] if block_index != 0 else None

            self.decoder_blocks.append(
                DecoderBlock(
                    num_spatial_dimensions,
                    num_inner_features * multiplier + (num_inner_features * prev_multiplier if prev_multiplier is not None else num_inner_features * multiplier),
                    num_inner_features * multiplier,
                    num_time_embedding_dims=num_inner_features,
                    num_rb_blocks=num_rb_blocks,
                    num_rb_conv_blocks=num_rb_conv_blocks,
                    self_attention=reversed_attention_levels[block_index],
                    num_attention_heads=num_attention_heads,
                    num_groups=num_groups,
                    kernel_size=3, stride=1, padding=1
                )
            )

        # Output Block
        self.output = nn.Sequential(
            ConvolutionalBlock(
                num_spatial_dimensions,
                num_inner_features + num_inner_features * reversed_feature_multipliers[-1],
                num_inner_features,
                num_groups=num_groups,
                activation=True,
                max_pool=False,
                kernel_size=3, stride=1, padding=1
            ),
            ConvolutionalBlock(
                num_spatial_dimensions,
                num_inner_features,
                out_channels,
                num_groups=0,
                activation=False,
                max_pool=False,
                kernel_size=1, stride=1, padding=0
            )
        )

    def initialize_weights(self, scale=1.0):
        for module in self.modules():
            if isinstance(module, (nn.Conv3d if self.num_spatial_dimensions == 3 else nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight.data)
                module.weight.data *= scale
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0)
            elif isinstance(module, nn.GroupNorm):
                nn.init.constant_(module.weight.data, 1)
                nn.init.constant_(module.bias.data, 0)

    def forward(self, x, x_cond, timestep):
        assert torch.all((torch.tensor(x.shape) % torch.tensor(x_cond.shape))[-self.num_spatial_dimensions:] == 0), f'Input spatial shape ({tuple(x.shape[-self.num_spatial_dimensions:])}) must be divisible by conditioning variable spatial shape ({tuple(x_cond.shape[-self.num_spatial_dimensions:])}) in every dimension'

        # Pad input and conditioning variable to the nearest multiple
        pad_multiple = 2 ** len(self.encoder_blocks)

        x_pad = (pad_multiple - (torch.tensor(x.shape[-self.num_spatial_dimensions:]) % pad_multiple)) % pad_multiple
        x_cond_pad = x_pad // (torch.tensor(x.shape) // torch.tensor(x_cond.shape))[-self.num_spatial_dimensions:]

        x = nn.functional.pad(x, pad=list(reversed(torch.stack((x_pad, torch.zeros_like(x_pad)), dim=1).flatten())), mode='reflect')
        x_cond = nn.functional.pad(x_cond, pad=list(reversed(torch.stack((x_cond_pad, torch.zeros_like(x_cond_pad)), dim=1).flatten())), mode='reflect')

        # Compute timestep embedding using sinusoidal positional encoding
        t_cond = self.time_positional_encoding(timestep)

        # Apply multilayer perceptron
        t_cond = self.time_mlp(t_cond)

        # Upscale conditioning variable
        x_cond_up = self.upsample_cond(x_cond)

        # Apply input layer transformation
        out = self.input(x)

        # Add conditioning variable
        out = out + x_cond_up

        skip_connection = out
        skip_connections = []

        # Contracting Path
        for encoder_block in self.encoder_blocks:
            out = encoder_block(out, t_cond)

            skip_connections.append(out)

        # Bottleneck
        out = self.bottleneck_block(out, t_cond)

        # Expanding Path
        for encoder_block_out, decoder_block in zip(reversed(skip_connections), self.decoder_blocks):
            out = torch.cat([out, encoder_block_out], dim=1)
            out = decoder_block(out, t_cond)

        # Concat skip connection and apply output layer
        out = torch.cat([out, skip_connection], dim=1)
        out = self.output(out)

        # Unpad output
        slicer = [slice(None)] * (out.dim() - self.num_spatial_dimensions) + [slice(None, -pad if pad > 0 else None) for pad in x_pad]
        out = out[slicer]

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
        noise_predictor_inferer.clear_conditions_cache()

        return x_t


if __name__ == '__main__':
    pass
