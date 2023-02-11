from e2e_sr import ConvolutionalBlock as SuperResolutionConvolutionBlock
from e2e_sr import UpsampleBlock as SuperResolutionUpsampleBlock
from e2e_seg import ConvolutionalBlock as SegmentationConvolutionBlock
from e2e_seg import UpsampleBlock as SegmentationUpsampleBlock


# VER ESTE CODIGO EXEMPLO QUE ENSINA A FAZER SKIP-CONNECTIONS ENTRE SUB BLOCKS DE 2 MODULES.
# NOS QUEREMOS FAZER SKIP CONNECTIONS ENTRE SUB BLOCKS DO MODULE DE SR E DO MODULE DE SEG
# SERA UTIL?????
class SuperResolutionBlocks(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sub_block_a = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.sub_block_b = torch.nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.sub_block_a(x)
        x = self.sub_block_b(x)
        return x


class SegmentationBlocks(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sub_block_c = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.sub_block_d = torch.nn.BatchNorm2d(128)

    def forward(self, x, skip_inputs):
        x = self.sub_block_c(x)
        x = self.sub_block_d(x)
        x = x + skip_inputs
        return x


class AugmentedGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Super-Resolution Blocks
        self.sr_blocks = SuperResolutionBlocks()

        # Segmentation Blocks
        self.seg_blocks = SegmentationBlocks()

    def forward(self, x):
        x = self.sr_blocks(x)

        return self.seg_blocks(x, skip_inputs=x)






class E2ESRSEG:
    def __init__(self, skip_connections=True):
        super(E2ESRSEG, self).__init__()

        # FULL NETWORK
        self.augmented_generator = AugmentedGenerator(...)

        # Define U-Net
        self.unet = self.augmented_generator.unet(...)

        # Get generator
        self.generator = self.augmented_generator.generator(...)

        # Define Discriminator
        self.discriminator = Discriminator(...)

        # Losses
        self.criterion_GAN = ...
        self.criterion_DICE = ...

        # Optimizers
        # 4 (1 para full network, 1 para generator, 1 para unet, 1 para discriminator)

    def forward(self, x):
        return self.augmented_generator(x)
