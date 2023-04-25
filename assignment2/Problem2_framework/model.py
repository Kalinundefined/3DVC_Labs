import torch
from torch import nn
from torchvision.models.vgg import vgg19
from torchvision.models.squeezenet import squeezenet1_1
from torchvision.models.alexnet import alexnet

class Img2PcdModel(nn.Module):
    """
    A neural network of single image to 3D.
    """

    def __init__(self, device, feature_extractor='vgg19'):
        super(Img2PcdModel, self).__init__()
        # TODO: Design your network layers.

        # vgg = vgg19().features
        # print(vgg)
        # self.vgg = nn.Sequential()
        # r = 0
        # for l in vgg:
        #     self.vgg.add_module(str(r), l)
        #     r += 1
        #     if r >= 23:
        #         break
        # self.vgg.add_module(
        #     str(r), nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        # )
        # self.vgg.add_module(str(r + 1), nn.ReLU(inplace=True))
        # self.vgg.add_module(
        #     str(r + 2), nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        # )
        # self.vgg.add_module(str(r + 3), nn.ReLU(inplace=True))
        self.input_conv1 = nn.Conv2d(3, 3, kernel_size=1)
        self.input_conv2 = nn.Conv2d(3, 3, kernel_size=1)
        self.ReLU = nn.ReLU(inplace=True)
        self.output_conv =  nn.Conv2d(512, 1000, kernel_size=6)
        self.feature_extractor = feature_extractor
        if self.feature_extractor == 'squeeze_net':
            self.encoder = squeezenet1_1(pretrained=False).features
            self.mlp = nn.Linear(100, 3)
        elif self.feature_extractor == 'vgg19':
            self.encoder = vgg19(pretrained=False).features
            self.mlp = nn.Linear(9, 3)
        elif self.feature_extractor == 'alexnet':
            self.encoder = alexnet(pretrained=False).features
            self.output_conv = nn.Identity()
            self.mlp = nn.Linear(49, 3)
        # print(self.encoder)
        # TODO: add more decoders
        
        self.device = device
        self.to(device)

    def forward(self, x):  # shape = (B, 3, 256, 256)
        # TODO: Design your network computation process.
        # Example:
        
        x = self.input_conv1(x)
        x = self.ReLU(x)
        x = self.input_conv2(x)
        x = self.ReLU(x)
        x = self.encoder(x)
        x = self.output_conv(x)
        # print(x.shape)
        x = self.ReLU(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.mlp(x)

        return x