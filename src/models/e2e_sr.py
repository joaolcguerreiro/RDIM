import pytorch_lightning as pl
import torch
import torch.nn as nn
import math


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False, activation=True, max_pool=False, **kwargs):
        super(ConvolutionalBlock, self).__init__()

        # Convolutional Layer
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

        # Batch Normalization Layer
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()

        # Activation Function
        self.act = nn.LeakyReLU(negative_slope=0.2) if activation else nn.Identity()

        # Max Polling Layer
        self.pool = nn.MaxPool2d(kernel_size=2) if max_pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))


class UpsampleBlock(nn.Module):
    def __int__(self, in_channels, out_channels, activation=True, **kwargs):
        super(UpsampleBlock, self).__int__()

        # Upsample Layer
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Convolutional Block
        self.conv = ConvolutionalBlock(
            in_channels,
            out_channels,
            batch_norm=False,
            activation=activation,
            max_pool=False,
            **kwargs
        )

    def forward(self, x):
        return self.conv(self.upsample(x))


class ResidualDenseBlock(nn.Module):
    def __int__(self, in_channels, num_growth_channel):
        super(DenseResidualBlock, self).__int__()

        # Convolutional Blocks
        self.conv_blocks = nn.ModuleList()

        for block_index in range(5):
            self.conv_blocks.append(
                ConvolutionalBlock(
                    in_channels + block_index * num_growth_channel,
                    num_growth_channel if block_index != 4 else in_channels,
                    batch_norm=False, activation=(True if block_index != 4 else False), max_pool=False,
                    kernel_size=3, stride=1, padding=1
                )
            )

    def forward(self, x):
        skip_connections = x

        for conv_block in self.conv_blocks:
            out = conv_block(skip_connections)

            skip_connections = torch.cat([skip_connections, out], dim=1)

        return out


class ResidualInResidualDenseBlock(nn.Module):
    def __int__(self, in_channels, num_growth_channel, beta=0.2):
        super(ResidualInResidualDenseBlock, self).__int__()

        # Residual Scale Factor
        self.beta = beta

        # Residual Dense Blocks
        self.res_blocks = nn.ModuleList()

        for _ in range(3):
            self.res_blocks.append(
                ResidualDenseBlock(
                    in_channels,
                    num_growth_channel
                )
            )

    def forward(self, x):
        skip_connection = x

        for res_block in self.res_blocks:
            out = res_block(skip_connection)

            skip_connection = skip_connection + out * self.beta

        return skip_connection * self.beta + x


class Generator(nn.Module):
    def __int__(self, in_channels=3, out_channels=3, num_features=64, num_blocks=23, num_growth_channel=32, scale_factor=2):
        super(Generator, self).__int__()

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
            num_growth_channel
        ) for _ in range(num_blocks)])

        # Convolutional Layer
        self.conv = ConvolutionalBlock(
            num_features,
            num_features,
            batch_norm=False,
            activation=False,
            max_pool=False,
            kernel_size=3, stride=1, padding=1
        )

        # Upsampling Block
        self.upsample_blocks = nn.Sequential(*[UpsampleBlock(
            num_features,
            num_features,
            activation=True,
            kernel_size=3, stride=1, padding=1
        ) for _ in range(math.log2(scale_factor))])

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
                in_channels,
                batch_norm=False,
                activation=False,
                max_pool=False,
                kernel_size=3, stride=1, padding=1
            )
        )

    def forward(self, x):
        x = skip_connection = self.input(x)

        x = self.res_in_res_blocks(x)

        x = self.conv(x) + skip_connection

        x = self.upsample_blocks(x)

        return self.output(x)


class Discriminator(nn.Module):
    def __int__(self, in_channels=3, num_features=64, num_blocks=8, blocks_output_size=6):
        super(Discriminator, self).__int__()

        # Convolutional Blocks
        self.conv_blocks = nn.Sequential(*[ConvolutionalBlock(
            num_features * 2 ** ((block_index - 1) // 2) if block_index != 0 else in_channels,
            num_features * 2 ** (block_index // 2),
            batch_norm=(True if block_index != 0 else False), activation=True, max_pool=False,
            kernel_size=3, stride=(1 + block_index % 2), padding=1
        ) for block_index in range(num_blocks)])

        # Output Block
        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d(blocks_output_size),
            nn.Flatten(),
            nn.Linear(num_features * (2 ** ((num_blocks - 1) // 2)) * blocks_output_size * blocks_output_size, 1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.conv_blocks(x)

        return self.output(x)


class E2ESR:
    def __init__(self):
        super().__int__()


if __name__ == '__main__':
    # net = Discriminator_VGG_patch_192(3, 64)
    # data = torch.randn(8, 3, 192, 192)
    # output = net(data)
    # print(output.shape)


