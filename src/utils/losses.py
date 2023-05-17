import torch.nn as nn
from torchvision .models import vgg19


class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()

        # Define Criterion
        self.criterion = nn.L1Loss()

        # VGG Network as feature extractor
        self.vgg = vgg19(pretrained=True).features[:35]

        # Set the model to eval mode
        self.vgg.eval()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        return self.criterion(self.vgg(input), self.vgg(target))
