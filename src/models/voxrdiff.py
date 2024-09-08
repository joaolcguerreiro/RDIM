import os
import sys
import torch
import torch.nn as nn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.utils import upsample


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, num_embedding_dims):
        super().__init__()

        self.num_embedding_dims = num_embedding_dims

    def forward(self, position):
        # Get indexes from even embedding dimensions
        even_indexes = torch.arange(0, self.num_embedding_dims, 2)

        # Compute positional embedding
        log_term = torch.tensor(1e4).log() / self.num_embedding_dims
        div_term = torch.exp(even_indexes * -log_term)

        position = position.unsqueeze(-1)
        sin_emb = torch.sin(position * div_term)
        cos_emb = torch.cos(position * div_term)

        # Interleave sine and cosine embeddings
        embedding = torch.stack([sin_emb, cos_emb], dim=-1).view(-1, 2 * ((self.num_embedding_dims + 1) // 2))

        # Slice out last dimension when the number of dimensions is uneven
        embedding = embedding[..., :self.num_embedding_dims]

        return embedding


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=32, activation=True, use_mish=True, max_pool=False, **kwargs):
        super().__init__()

        # Convolutional Layer
        self.conv = nn.Conv3d(in_channels, out_channels, **kwargs)

        # Group Normalization Layer
        self.gn = nn.GroupNorm(num_groups, out_channels) if num_groups > 0 else nn.Identity()

        # Activation Function
        self.act = (nn.Mish() if use_mish else nn.LeakyReLU(negative_slope=0.2)) if activation else nn.Identity()

        # Max Polling Layer
        self.pool = nn.MaxPool3d(kernel_size=2) if max_pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.act(self.gn(self.conv(x))))


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, activation=False, use_mish=True, **kwargs):
        super().__init__()

        # Convolutional Layer
        self.conv = nn.Conv3d(in_channels, out_channels, stride=scale_factor, padding_mode='reflect', **kwargs)

        # Activation Function
        self.act = (nn.Mish() if use_mish else nn.LeakyReLU(negative_slope=0.2)) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, mode='trilinear', activation=False, use_mish=True, **kwargs):
        super().__init__()

        # Upsample Layer
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=False)

        # Convolutional Layer
        self.conv = nn.Conv3d(in_channels, out_channels, **kwargs)

        # Activation Function
        self.act = (nn.Mish() if use_mish else nn.LeakyReLU(negative_slope=0.2)) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(self.upsample(x)))


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_time_embedding_dims, num_growth_features=32, num_conv_blocks=3, num_groups=32, beta=0.2, **kwargs):
        super().__init__()
        assert num_conv_blocks > 0 if in_channels != out_channels else True, f'Number of convolutional blocks ({num_conv_blocks}) must be greater than 0'

        # Residual Scale Factor
        self.beta = beta

        # Time embedding projection layer
        self.time_proj = nn.Sequential(
            nn.Mish(),
            nn.Linear(num_time_embedding_dims, num_growth_features)
        )

        # Convolutional Blocks
        self.conv_blocks = nn.ModuleList()

        for block_index in range(num_conv_blocks):
            self.conv_blocks.append(
                ConvolutionalBlock(
                    in_channels + block_index * num_growth_features,
                    num_growth_features if block_index != (num_conv_blocks - 1) else out_channels,
                    num_groups=(num_groups if block_index != (num_conv_blocks - 1) else 0),
                    activation=(True if block_index != (num_conv_blocks - 1) else False),
                    use_mish=True,
                    max_pool=False,
                    **(kwargs if block_index != (num_conv_blocks - 1) else {'kernel_size': 1, 'stride': 1, 'padding': 0})
                )
            )

        # Skip connection Layer
        self.residual_conv = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_cond):
        # Apply time embedding projection
        t_cond = self.time_proj(t_cond)

        # Count dimension mismatch between input tensor and timestep embedding
        num_diff_dims = x.dim() - t_cond.dim()

        # Reshape time embedding to match input tensor dimensions
        t_cond = t_cond.view(*t_cond.shape, *([1] * num_diff_dims))

        out = skip_connection = skip_connections = x

        for block_index, conv_block in enumerate(self.conv_blocks):
            out = conv_block(skip_connections)

            if block_index < len(self.conv_blocks) - 1:
                out = out + t_cond

            skip_connections = torch.cat([skip_connections, out], dim=1)

        out = out * self.beta + self.residual_conv(skip_connection)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_embedding_dims, num_heads=8):
        super().__init__()
        assert num_embedding_dims % num_heads == 0, f'Number of embedding dimensions ({num_embedding_dims}) must be divisible by the number of attention heads ({num_heads})'

        self.num_heads = num_heads
        self.num_head_dims = num_embedding_dims // num_heads

        # Linear projections for queries (Q), keys (K), and values (V)
        self.query_proj = nn.Linear(num_embedding_dims, num_embedding_dims)
        self.key_proj = nn.Linear(num_embedding_dims, num_embedding_dims)
        self.value_proj = nn.Linear(num_embedding_dims, num_embedding_dims)

        # Linear projection for the output after head concatenation
        self.output_proj = nn.Linear(num_embedding_dims, num_embedding_dims)

    def forward(self, x):
        # Get linear projections of queries (Q), keys (K), and values (V)
        # [batch, sequence_len, num_embedding_dims] => [batch, sequence_len, num_embedding_dims]
        queries = self.query_proj(x)
        keys = self.key_proj(x)
        values = self.value_proj(x)

        # Split queries (Q), keys (K), and values (V) between attention heads
        # [batch, sequence_len, num_embedding_dims] => [batch, num_heads, sequence_len, num_head_dims]
        queries = torch.stack(torch.split(queries, split_size_or_sections=self.num_head_dims, dim=-1), dim=1)
        keys = torch.stack(torch.split(keys, split_size_or_sections=self.num_head_dims, dim=-1), dim=1)
        values = torch.stack(torch.split(values, split_size_or_sections=self.num_head_dims, dim=-1), dim=1)

        # Compute Scaled Dot-Product Attention
        # Eq. 1 from Transformer paper (https://arxiv.org/pdf/1706.03762.pdf)
        scale_factor = 1 / torch.tensor(self.num_head_dims).sqrt()
        transposed_keys = keys.transpose(-2, -1).contiguous()
        attention_scores = torch.matmul(torch.softmax(torch.matmul(queries, transposed_keys) * scale_factor, dim=-1), values)

        # Permute dimensions of Attention Scores
        # [batch, num_heads, sequence_len, num_head_dims] => [batch, sequence_len, num_heads, num_head_dims]
        attention_scores = attention_scores.permute(0, 2, 1, 3).contiguous()

        # Concatenate heads
        # [batch, sequence_len, num_heads, num_head_dims] => [batch, sequence_len, num_embedding_dims]
        concatenated_heads_attention_scores = attention_scores.view(*x.shape)

        # Get output linear projection
        out = self.output_proj(concatenated_heads_attention_scores)

        return out


class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, num_heads=8, num_groups=32):
        super().__init__()

        # Multi-Head Attention Mechanism
        self.attention = MultiHeadAttention(
            num_embedding_dims=in_channels,
            num_heads=num_heads
        )

        # Group Normalization Layer
        self.gn = nn.GroupNorm(num_groups, in_channels) if num_groups > 0 else nn.Identity()

    def forward(self, x):
        skip_connection = x

        # Reshape and transpose input tensor to process depth feature maps rather than spatial maps
        batch, in_channels, *spatial_shape = x.shape

        # [batch, in_channels, D, H, W] => [batch, D*H*W, in_channels] = [batch, sequence_len, num_embedding_dims]
        x = x.view(batch, in_channels, torch.tensor(spatial_shape).prod()).transpose(-2, -1).contiguous()

        # Apply Attention Mechanism
        out = self.attention(x)

        # Transpose and reshape output tensor to match input shape
        # [batch, D*H*W, in_channels] => [batch, in_channels, D, H, W]
        out = out.transpose(-2, -1).contiguous().view(batch, in_channels, *spatial_shape)

        # Add skip-connection and apply normalization
        out = self.gn(out + skip_connection)

        return out


class ResidualDenseAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_time_embedding_dims, num_rdb_growth_features=32, num_rdb_conv_blocks=3, self_attention=True, num_attention_heads=8, num_groups=32, **kwargs):
        super().__init__()

        # Residual Dense Block
        self.res_dense_block = ResidualDenseBlock(
            in_channels,
            out_channels,
            num_time_embedding_dims=num_time_embedding_dims,
            num_growth_features=num_rdb_growth_features,
            num_conv_blocks=num_rdb_conv_blocks,
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
        out = self.res_dense_block(x, t_cond)

        out = self.att_block(out)

        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_time_embedding_dims, num_rdb_growth_features=32, num_rdb_blocks=1, num_rdb_conv_blocks=3, self_attention=True, num_attention_heads=8, num_groups=32, **kwargs):
        super().__init__()

        # Residual Dense Blocks with Optional Self-Attention Blocks
        self.res_dense_att_blocks = nn.ModuleList()

        for block_index in range(num_rdb_blocks):
            self.res_dense_att_blocks.append(
                ResidualDenseAttentionBlock(
                    out_channels if block_index != 0 else in_channels,
                    out_channels,
                    num_time_embedding_dims=num_time_embedding_dims,
                    num_rdb_growth_features=num_rdb_growth_features,
                    num_rdb_conv_blocks=num_rdb_conv_blocks,
                    self_attention=self_attention,
                    num_attention_heads=num_attention_heads,
                    num_groups=num_groups,
                    **kwargs
                )
            )

        # Downsampling Block
        self.downsample_block = DownsampleBlock(
            out_channels,
            out_channels,
            scale_factor=2,
            activation=False,
            kernel_size=kwargs.get('kernel_size', 3), padding=kwargs.get('padding', 1)
        )

    def forward(self, x, t_cond):
        out = x

        for res_dense_att_block in self.res_dense_att_blocks:
            out = res_dense_att_block(out, t_cond)

        out = self.downsample_block(out)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_time_embedding_dims, num_rdb_growth_features=32, num_rdb_blocks=1, num_rdb_conv_blocks=3, self_attention=True, num_attention_heads=8, num_groups=32, **kwargs):
        super().__init__()

        # Residual Dense Blocks with Optional Self-Attention Blocks
        self.res_dense_att_blocks = nn.ModuleList()

        for block_index in range(num_rdb_blocks):
            self.res_dense_att_blocks.append(
                ResidualDenseAttentionBlock(
                    out_channels if block_index != 0 else in_channels,
                    out_channels,
                    num_time_embedding_dims=num_time_embedding_dims,
                    num_rdb_growth_features=num_rdb_growth_features,
                    num_rdb_conv_blocks=num_rdb_conv_blocks,
                    self_attention=self_attention,
                    num_attention_heads=num_attention_heads,
                    num_groups=num_groups,
                    **kwargs
                )
            )

        # Upsampling Block
        self.upsample_block = UpsampleBlock(
            out_channels,
            out_channels,
            scale_factor=2,
            mode='trilinear',
            activation=False,
            **kwargs
        )

    def forward(self, x, t_cond):
        out = x

        for res_dense_att_block in self.res_dense_att_blocks:
            out = res_dense_att_block(out, t_cond)

        out = self.upsample_block(out)

        return out


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, num_time_embedding_dims, num_rdb_growth_features=32, num_rdb_blocks=1, num_rdb_conv_blocks=3, self_attention=True, num_attention_heads=8, num_groups=32, **kwargs):
        super().__init__()

        # Residual Dense Blocks with Optional Self-Attention Blocks
        self.res_dense_att_blocks = nn.ModuleList()

        for _ in range(num_rdb_blocks):
            self.res_dense_att_blocks.append(
                ResidualDenseAttentionBlock(
                    in_channels,
                    in_channels,
                    num_time_embedding_dims=num_time_embedding_dims,
                    num_rdb_growth_features=num_rdb_growth_features,
                    num_rdb_conv_blocks=num_rdb_conv_blocks,
                    self_attention=self_attention,
                    num_attention_heads=num_attention_heads,
                    num_groups=num_groups,
                    **kwargs
                )
            )

    def forward(self, x, t_cond):
        out = x

        for res_dense_att_block in self.res_dense_att_blocks:
            out = res_dense_att_block(out, t_cond)

        return out


class NoisePredictor(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_cond_features=64, num_inner_features=64, feature_multipliers=(1, 2, 4, 8), num_rdb_blocks=1, num_rdb_conv_blocks=3, num_rdb_growth_features=32, attention_levels=((False,) * 2 + (True,) * 2), num_attention_heads=8, num_groups=32, scale_factor=1):
        super().__init__()
        assert len(feature_multipliers) == len(attention_levels), f'Array size of attention levels ({len(attention_levels)}) must match array size of feature multipliers ({len(feature_multipliers)})'

        # Positional Encoding
        self.time_positional_encoding = SinusoidalPositionalEncoding(num_inner_features)

        # Multilayer Perceptron for time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(num_inner_features, num_inner_features * 4),
            nn.Mish(),
            nn.Linear(num_inner_features * 4, num_inner_features)
        )

        # Upsample/Projection block for condition
        self.upsample_cond = UpsampleBlock(
            num_cond_features,
            num_inner_features,
            scale_factor=scale_factor,
            mode='trilinear',
            activation=False,
            kernel_size=3, stride=1, padding=1
        ) if scale_factor > 1 or num_cond_features != num_inner_features else nn.Identity()

        # Input Layer
        self.input = ConvolutionalBlock(
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
                    num_inner_features * prev_multiplier if prev_multiplier is not None else num_inner_features,
                    num_inner_features * multiplier,
                    num_time_embedding_dims=num_inner_features,
                    num_rdb_growth_features=num_rdb_growth_features,
                    num_rdb_blocks=num_rdb_blocks,
                    num_rdb_conv_blocks=num_rdb_conv_blocks,
                    self_attention=attention_levels[block_index],
                    num_attention_heads=num_attention_heads,
                    num_groups=num_groups,
                    kernel_size=3, stride=1, padding=1
                )
            )

        # Bottleneck block
        self.bottleneck_block = BottleneckBlock(
            num_inner_features * feature_multipliers[-1],
            num_time_embedding_dims=num_inner_features,
            num_rdb_growth_features=num_rdb_growth_features,
            num_rdb_blocks=num_rdb_blocks,
            num_rdb_conv_blocks=num_rdb_conv_blocks,
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
                    num_inner_features * multiplier + (num_inner_features * prev_multiplier if prev_multiplier is not None else num_inner_features * multiplier),
                    num_inner_features * multiplier,
                    num_time_embedding_dims=num_inner_features,
                    num_rdb_growth_features=num_rdb_growth_features,
                    num_rdb_blocks=num_rdb_blocks,
                    num_rdb_conv_blocks=num_rdb_conv_blocks,
                    self_attention=reversed_attention_levels[block_index],
                    num_attention_heads=num_attention_heads,
                    num_groups=num_groups,
                    kernel_size=3, stride=1, padding=1
                )
            )

        # Output Block
        self.output = nn.Sequential(
            ConvolutionalBlock(
                num_inner_features + num_inner_features * reversed_feature_multipliers[-1],
                num_inner_features,
                num_groups=num_groups,
                activation=True,
                use_mish=True,
                max_pool=False,
                kernel_size=3, stride=1, padding=1
            ),
            ConvolutionalBlock(
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
            if isinstance(module, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight.data)
                module.weight.data *= scale
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0)
            elif isinstance(module, nn.GroupNorm):
                nn.init.constant_(module.weight.data, 1)
                nn.init.constant_(module.bias.data, 0)

    def forward(self, x, x_cond, timestep):
        # Compute timestep embedding using sinusoidal positional encoding
        t_cond = self.time_positional_encoding(timestep)

        # Apply multilayer perceptron
        t_cond = self.time_mlp(t_cond)

        # Upscale conditioning variable
        x_cond_up = self.upsample_cond(x_cond)

        # Apply input layer transformation
        x = self.input(x)

        # Add conditioning variable
        x = x + x_cond_up

        out = skip_connection = x
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

        # Concat skip-connection and apply output layer
        out = torch.cat([out, skip_connection], dim=1)
        out = self.output(out)

        return out


class GaussianDiffusion(nn.Module):
    def __init__(self, num_timesteps=1000, beta_schedule='COSINE', residual_prediction=True, lr_condition_encoding=True, scale_factor=2, **kwargs):
        super().__init__()

        self.num_timesteps = num_timesteps
        self.betas = self.compute_beta_schedule(beta_schedule=beta_schedule, num_timesteps=num_timesteps, **kwargs)
        self.alphas = 1.0 - self.betas
        self.alphas_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.residual_prediction = residual_prediction
        self.lr_condition_encoding = lr_condition_encoding
        self.scale_factor = scale_factor

    @staticmethod
    def linear_beta_schedule(num_timesteps=1000, beta_start=1e-4, beta_end=2e-2):
        """
        Linear schedule, proposed in original DDPM paper (https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)
        """

        return torch.linspace(start=beta_start, end=beta_end, steps=num_timesteps)

    @staticmethod
    def cosine_beta_schedule(num_timesteps=1000, beta_offset=8e-3, beta_min=0.0, beta_max=9.99e-1):
        """
        Cosine schedule, proposed in improved DDPM paper (https://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf)
        """

        steps = torch.linspace(start=0, end=num_timesteps, steps=(num_timesteps + 1))
        cosine_function = torch.cos((steps / num_timesteps + beta_offset) / (1 + beta_offset) * (torch.pi / 2)) ** 2
        alphas_cum_prod = cosine_function / cosine_function[0]
        betas = 1 - (alphas_cum_prod[1:] / alphas_cum_prod[:-1])

        return torch.clamp(betas, min=beta_min, max=beta_max)

    @staticmethod
    def sigmoid_beta_schedule(num_timesteps=1000, start=-3, end=3, tau=1, beta_min=0.0, beta_max=9.99e-1):
        """
        Sigmoid schedule, proposed in RINs paper (https://arxiv.org/pdf/2212.11972.pdf)
        """

        steps = torch.linspace(start=0, end=num_timesteps, steps=(num_timesteps + 1))
        v_start = torch.tensor(start / tau).sigmoid()
        v_end = torch.tensor(end / tau).sigmoid()
        sigmoid_function = (-(((steps / num_timesteps) * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
        alphas_cum_prod = sigmoid_function / sigmoid_function[0]
        betas = 1 - (alphas_cum_prod[1:] / alphas_cum_prod[:-1])

        return torch.clamp(betas, min=beta_min, max=beta_max)

    def compute_beta_schedule(self, beta_schedule='COSINE', num_timesteps=1000, **kwargs):
        if beta_schedule == 'LINEAR':
            return self.linear_beta_schedule(num_timesteps=num_timesteps, **kwargs)
        elif beta_schedule == 'COSINE':
            return self.cosine_beta_schedule(num_timesteps=num_timesteps, **kwargs)
        elif beta_schedule == 'SIGMOID':
            return self.sigmoid_beta_schedule(num_timesteps=num_timesteps, **kwargs)
        else:
            raise NotImplementedError(f'Beta schedule {beta_schedule} not implemented.')

    def compute_volume_at_timestep(self, x_0, timestep):
        """
        Compute the volume at a specific timestep given the initial volume
        Equation (9) in improved DDPM paper (https://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf)
        """

        assert torch.all((timestep >= 0) & (timestep <= self.num_timesteps)), f'Timestep(s) ({timestep}) must be (a) greater than or equal to 0 and (b) less than or equal to ({self.num_timesteps})'

        # Define condition to check if timestep is greater than 0
        condition = timestep > 0

        # Get timestep index, which equals to -1 when timestep is 0 (initial), thus index errors are absent
        timestep_index = timestep - 1

        # Count dimension mismatch between volume and timestep
        num_diff_dims = x_0.dim() - timestep.dim()

        # Compute alpha cumulative product at given timestep and match volume dimensions
        alpha_cum_prod_t = torch.where(condition, self.alphas_cum_prod[timestep_index], float('nan')).view(*timestep.shape, *([1] * num_diff_dims))

        condition = condition.view(*timestep.shape, *([1] * num_diff_dims))

        # Sample random noise
        eps = torch.randn_like(x_0)

        x_t = torch.where(condition, torch.sqrt(alpha_cum_prod_t) * x_0 + torch.sqrt(1 - alpha_cum_prod_t) * eps, x_0)

        return x_t

    def compute_volume_at_next_timestep(self, x_t_m1, timestep):
        """
        Compute the volume at the next timestep given the volume at the previous timestep
        Equation (2) in original DDPM paper (https://arxiv.org/pdf/2006.11239.pdf) with reparameterization trick
        """

        assert torch.all((timestep >= 1) & (timestep <= self.num_timesteps)), f'Timestep(s) ({timestep}) must be (a) greater than or equal to 1 and (b) less than or equal to ({self.num_timesteps})'

        # Get timestep index
        timestep_index = timestep - 1

        # Count dimension mismatch between volume and timestep
        num_diff_dims = x_t_m1.dim() - timestep.dim()

        # Compute beta at given timestep and match volume dimensions
        beta_t = self.betas[timestep_index].view(*timestep.shape, *([1] * num_diff_dims))

        # Sample random noise
        eps_t = torch.randn_like(x_t_m1)

        x_t = torch.sqrt(1 - beta_t) * x_t_m1 + torch.sqrt(beta_t) * eps_t

        return x_t, eps_t

    def compute_volume_at_prev_timestep(self, x_t, eps_t, timestep):
        """
        Compute the volume at the previous timestep given the volume at the precedent timestep
        Line (4) in Sampling algorithm of original DDPM paper (https://arxiv.org/pdf/2006.11239.pdf)
        """

        assert torch.all((timestep >= 1) & (timestep <= self.num_timesteps)), f'Timestep(s) ({timestep}) must be (a) greater than or equal to 1 and (b) less than or equal to ({self.num_timesteps})'

        # Define condition to check if timestep is greater than 1
        condition = timestep > 1

        # Get timestep index
        timestep_index = timestep - 1

        # Count dimension mismatch between volume and timestep
        num_diff_dims = x_t.dim() - timestep.dim()

        # Compute diffusion parameters and match volume dimensions
        beta_t = self.betas[timestep_index].view(*timestep.shape, *([1] * num_diff_dims))
        alpha_t = self.alphas[timestep_index].view(*timestep.shape, *([1] * num_diff_dims))
        alpha_cum_prod_t = self.alphas_cum_prod[timestep_index].view(*timestep.shape, *([1] * num_diff_dims))

        condition = condition.view(*timestep.shape, *([1] * num_diff_dims))

        # Sample random noise
        z = torch.where(condition, torch.randn_like(x_t), torch.zeros_like(x_t))

        x_t_m1 = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / (torch.sqrt(1 - alpha_cum_prod_t))) * eps_t) + torch.sqrt(beta_t) * z

        return x_t_m1

    def sample_timesteps(self, num_samples):
        return torch.randint(low=1, high=self.num_timesteps + 1, size=(num_samples,), dtype=torch.long)

    def forward_diffusion(self, x_0, timestep):
        assert torch.all((timestep >= 1) & (timestep <= self.num_timesteps)), f'Timestep(s) ({timestep}) must be (a) greater than or equal to 1 and (b) less than or equal to ({self.num_timesteps})'

        # Get second to last timestep (timestep_m1)
        timestep_m1 = timestep - 1

        # Compute second to last timestep volume
        x_t_m1 = self.compute_volume_at_timestep(x_0, timestep_m1)

        # Compute last timestep volume
        x_t, eps_t = self.compute_volume_at_next_timestep(x_t_m1, timestep)

        return x_t, x_t_m1, eps_t

    def reverse_diffusion(self, x_t, eps_t, timestep):
        assert torch.all((timestep >= 1) & (timestep <= self.num_timesteps)), f'Timestep(s) ({timestep}) must be (a) greater than or equal to 1 and (b) less than or equal to ({self.num_timesteps})'

        # Predict previous timestep volume
        x_t_m1_pred = self.compute_volume_at_prev_timestep(x_t, eps_t, timestep)

        return x_t_m1_pred

    @torch.no_grad()
    def sample(self, x_lr, noise_predictor, lr_encoder=None, noise_predictor_inferer=None, lr_encoder_inferer=None):
        # Get number of samples (batch size)
        num_samples = x_lr.shape[0]

        # Set the noise predictor to evaluation mode
        noise_predictor.eval()

        # Prepare conditioning variable
        if self.lr_condition_encoding:
            # Set the LR encoder to evaluation mode
            lr_encoder.eval()

            x_cond = lr_encoder_inferer(inputs=x_lr, network=lr_encoder) if lr_encoder_inferer else lr_encoder(x_lr)
        else:
            x_cond = x_lr

        # Upsample volume
        x_lr_up = torch.stack([upsample(volume, self.scale_factor) for volume in x_lr])

        # Compute random noise
        x_t = torch.randn_like(x_lr_up)

        for timestep in reversed(range(1, self.num_timesteps + 1)):
            # Adapt timestep for batch processing
            timestep = torch.full((num_samples,), timestep, dtype=torch.long)

            # Predict noise
            eps_t_pred = noise_predictor_inferer(inputs=x_t, conditions=x_cond, timesteps=timestep, network=noise_predictor) if noise_predictor_inferer else noise_predictor(x_t, x_cond, timestep)

            # Reverse diffusion to compute volume at previous timestep
            x_t = self.reverse_diffusion(x_t, eps_t_pred, timestep)

        return x_t + x_lr_up if self.residual_prediction else x_t


if __name__ == '__main__':
    pass
