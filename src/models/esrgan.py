import torch
import torch.nn as nn
import math


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False, activation=True, max_pool=False, **kwargs):
        super().__init__()

        # Convolutional Layer
        self.conv = nn.Conv3d(in_channels, out_channels, **kwargs)

        # Batch Normalization Layer
        self.bn = nn.BatchNorm3d(out_channels) if batch_norm else nn.Identity()

        # Activation Function
        self.act = nn.LeakyReLU(negative_slope=0.2) if activation else nn.Identity()

        # Max Polling Layer
        self.pool = nn.MaxPool3d(kernel_size=2) if max_pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, mode='nearest', activation=True, **kwargs):
        super().__init__()

        # Upsample Layer
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)

        # Convolutional Layer
        self.conv = nn.Conv3d(in_channels, out_channels, **kwargs)

        # Activation Function
        self.act = nn.LeakyReLU(negative_slope=0.2) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(self.upsample(x)))


class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_growth_features=32, num_conv_blocks=5, **kwargs):
        super().__init__()

        # Convolutional Blocks
        self.conv_blocks = nn.ModuleList()

        for block_index in range(num_conv_blocks):
            self.conv_blocks.append(
                ConvolutionalBlock(
                    in_channels + block_index * num_growth_features,
                    num_growth_features if block_index != (num_conv_blocks - 1) else in_channels,
                    batch_norm=False,
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
    def __init__(self, in_channels, num_db_growth_features=32, num_db_blocks=3, num_db_conv_blocks=5, beta=0.2, **kwargs):
        super().__init__()

        # Residual Scale Factor
        self.beta = beta

        # Dense Blocks
        self.dense_blocks = nn.ModuleList()

        for _ in range(num_db_blocks):
            self.dense_blocks.append(
                DenseBlock(
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


class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=64, num_rrdb_blocks=23, num_rrdb_db_blocks=3, num_rrdb_db_conv_blocks=5, num_rrdb_db_growth_features=32, scale_factor=2):
        super().__init__()

        # Input Layer
        self.input = ConvolutionalBlock(
            in_channels,
            num_features,
            batch_norm=False,
            activation=False,
            max_pool=False,
            kernel_size=3, stride=1, padding=1
        )

        # Residual in Residual Dense Blocks
        self.res_in_res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(
            num_features,
            num_db_growth_features=num_rrdb_db_growth_features,
            num_db_blocks=num_rrdb_db_blocks,
            num_db_conv_blocks=num_rrdb_db_conv_blocks,
            kernel_size=3, stride=1, padding=1
        ) for _ in range(num_rrdb_blocks)])

        # Convolutional Layer
        self.conv = ConvolutionalBlock(
            num_features,
            num_features,
            batch_norm=False,
            activation=False,
            max_pool=False,
            kernel_size=3, stride=1, padding=1
        )

        # Upsampling Blocks
        self.upsample_blocks = nn.Sequential(*[UpsampleBlock(
            num_features,
            num_features,
            scale_factor=2,
            mode='nearest',
            activation=True,
            kernel_size=3, stride=1, padding=1
        ) for _ in range(int(math.log2(scale_factor)))])

        # Output Block
        self.output = nn.Sequential(
            ConvolutionalBlock(
                num_features,
                num_features,
                batch_norm=False,
                activation=True,
                max_pool=False,
                kernel_size=3, stride=1, padding=1
            ),
            ConvolutionalBlock(
                num_features,
                out_channels,
                batch_norm=False,
                activation=False,
                max_pool=False,
                kernel_size=3, stride=1, padding=1
            )
        )

    def initialize_weights(self, scale=1.0):
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight.data)
                module.weight.data *= scale
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0)

    def extract_features(self, x):
        out = skip_connection = self.input(x)

        out = self.res_in_res_blocks(out)

        features = self.conv(out) + skip_connection

        return features

    def forward(self, x):
        out = skip_connection = self.input(x)

        out = self.res_in_res_blocks(out)

        out = self.conv(out) + skip_connection

        out = self.upsample_blocks(out)

        out = self.output(out)

        return out


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, num_features=64, num_conv_blocks=8, conv_blocks_output_size=6):
        super().__init__()

        # Convolutional Blocks
        self.conv_blocks = nn.Sequential(*[ConvolutionalBlock(
            num_features * 2 ** ((block_index - 1) // 2) if block_index != 0 else in_channels,
            num_features * 2 ** (block_index // 2),
            batch_norm=(True if block_index != 0 else False),
            activation=True,
            max_pool=False,
            kernel_size=3, stride=(1 + block_index % 2), padding=1
        ) for block_index in range(num_conv_blocks)])

        # Output Block
        self.output = nn.Sequential(
            nn.AdaptiveAvgPool3d(conv_blocks_output_size),
            nn.Flatten(),
            nn.Linear(num_features * (2 ** ((num_conv_blocks - 1) // 2)) * (conv_blocks_output_size ** 3), 1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(1024, 1)
        )

    def initialize_weights(self, scale=1.0):
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight.data)
                module.weight.data *= scale
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0)
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight.data, 1)
                nn.init.constant_(module.bias.data, 0)

    def extract_features(self, x):
        features = self.conv_blocks(x)

        return features

    def forward(self, x):
        out = self.conv_blocks(x)

        out = self.output(out)

        return out


if __name__ == '__main__':
    pass
