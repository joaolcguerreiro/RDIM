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

        for parameter in self.vgg.parameters():
            parameter.requires_grad = False

    def forward(self, prediction, ground_truth):
        return self.criterion(self.vgg(prediction), self.vgg(ground_truth))
