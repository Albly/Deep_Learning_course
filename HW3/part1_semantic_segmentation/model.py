import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
import numpy as np
from torch.nn import Conv2d, ReLU, BatchNorm2d, Dropout, ConvTranspose2d, MaxPool2d


class DownBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()

        self.block = nn.Sequential(
            Conv2d(in_channels = in_ch , out_channels = out_ch, kernel_size = 3, stride = 1, padding = 1),
            BatchNorm2d(out_ch),
            ReLU(), 

            Conv2d(in_channels = out_ch , out_channels = out_ch , kernel_size = 3, stride = 1, padding = 1),
            BatchNorm2d(out_ch),
            ReLU()
        )
        self.pool  = MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
    
    def forward(self, x):
        x = self.block(x)
        return self.pool(x), x 


class BaseBlock(torch.nn.Module):
    def __init__(self, in_ch):
        super(BaseBlock, self).__init__()
        
        self.block = torch.nn.Sequential(
            Conv2d(in_channels=in_ch, out_channels = in_ch, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(in_ch),
            ReLU(),
            Conv2d(in_channels=in_ch, out_channels = in_ch, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(in_ch),
            ReLU()
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(torch.nn.Module):
    def __init__(self, in_ch , out_ch):
        super(UpBlock, self).__init__()

        self.up = ConvTranspose2d(in_channels = in_ch, out_channels= in_ch, kernel_size = 3, padding = 1, output_padding=1 , stride = 2)

        self.block = nn.Sequential(
            Dropout(p = 0.5),
            Conv2d(in_channels = in_ch*2, out_channels = out_ch, kernel_size = 3, stride= 1, padding=1),
            BatchNorm2d(out_ch),
            ReLU(),

            Conv2d(in_channels = out_ch, out_channels = out_ch, kernel_size = 3, stride= 1, padding=1),
            BatchNorm2d(out_ch),
            ReLU()
        )
        

    def forward(self, x, skip):
        x = self.up(x) 
        x = self.block(torch.cat((skip, x), dim = 1))

        return x 


class UNet(nn.Module):
    """
    TODO: 8 points

    A standard UNet network (with padding in covs).

    For reference, see the scheme in materials/unet.png
    - Use batch norm between conv and relu
    - Use max pooling for downsampling
    - Use conv transpose with kernel size = 3, stride = 2, padding = 1, and output padding = 1 for upsampling
    - Use 0.5 dropout after concat

    Args:
      - num_classes: number of output classes
      - min_channels: minimum number of channels in conv layers
      - max_channels: number of channels in the bottleneck block
      - num_down_blocks: number of blocks which end with downsampling

    The full architecture includes downsampling blocks, a bottleneck block and upsampling blocks

    You also need to account for inputs which size does not divide 2**num_down_blocks:
    interpolate them before feeding into the blocks to the nearest size which divides 2**num_down_blocks,
    and interpolate output logits back to the original shape
    """
    def __init__(self, 
                 num_classes,
                 min_channels=32,
                 max_channels=512, 
                 num_down_blocks=4):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.num_down_blocks = num_down_blocks  
        shapes = np.round(np.geomspace(min_channels,max_channels,num_down_blocks+1)).astype(int).tolist()

        self.in_layer = Conv2d(3,min_channels, 1)

        self.downblocks = nn.ModuleList([])
        for i in range(num_down_blocks):
            self.downblocks.append(DownBlock(in_ch = shapes[i], out_ch = shapes[i+1]))


        self.baseBlock = BaseBlock(max_channels)


        self.upBlocks = nn.ModuleList([])
        shapes.reverse()
        for i in range(num_down_blocks):
            self.upBlocks.append(UpBlock(in_ch = shapes[i], out_ch = shapes[i+1]))
        
        self.out_layer = Conv2d(min_channels , self.num_classes, 1)


    def forward(self, inputs):
        
        shape_h, shape_w = inputs.shape[2],inputs.shape[3]
        new_shape_h = (shape_h//2**(self.num_down_blocks))*2**(self.num_down_blocks)
        new_shape_w = (shape_w//2**(self.num_down_blocks))*2**(self.num_down_blocks)
        self.squeeze = nn.Upsample(size = (new_shape_h, new_shape_w), mode= 'bilinear', align_corners=False)
        self.unsqueeze = nn.Upsample(size = (shape_h, shape_w), mode= 'bilinear', align_corners=False)

        x = inputs
        skips = []

        x = self.squeeze(x)
        x = self.in_layer(x)


        for downblock in self.downblocks:
            x, skip = downblock(x)
            skips.append(skip)

        x = self.baseBlock(x)


        skips.reverse()
        for upblock, skip in zip(self.upBlocks, skips):

            x = upblock(x,skip)
        
        x = self.out_layer(x)

        logits = self.unsqueeze(x)

        #logits = None # TODO

        assert logits.shape == (inputs.shape[0], self.num_classes, inputs.shape[2], inputs.shape[3]), 'Wrong shape of the logits'
        return logits



# DEEPLAB 
# ========================================================================================================

class BlockConv2d(nn.Sequential):
    def __init__(self,in_channels, out_channels, kernel_size):
        super(BlockConv2d, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, bias= False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

class AtrousConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size ,dilation=1 ):
        super(AtrousConv2d, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=dilation, dilation = dilation , bias= False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    

class GlobalPool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GlobalPool,self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels = in_channels, out_channels= out_channels, kernel_size= 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
    def forward(self,x):
        h,w = x.shape[2], x.shape[3]
        self.ups = nn.Upsample(size = (h,w), mode= 'bilinear', align_corners=False)
        return self.ups(self.module(x))


class DeepLab(nn.Module):
    """
    TODO: 6 points

    (simplified) DeepLab segmentation network.
    
    Args:
      - backbone: ['resnet18', 'vgg11_bn', 'mobilenet_v3_small'],
      - aspp: use aspp module
      - num classes: num output classes

    During forward pass:
      - Pass inputs through the backbone to obtain features
      - Apply ASPP (if needed)
      - Apply head
      - Upsample logits back to the shape of the inputs
    """
    def __init__(self, backbone, aspp, num_classes):
        super(DeepLab, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.init_backbone()
        if aspp:
            self.aspp = ASPP(self.out_features, 256, [12, 24, 36])

        self.head = DeepLabHead(self.out_features, num_classes)

    def init_backbone(self):
        # TODO: initialize an ImageNet-pretrained backbone
        if self.backbone == 'resnet18':
            self.back = models.resnet18()
            self.back = nn.Sequential(*list(self.back.children())[:-2])
            self.out_features = 512 # TODO: number of output features in the backbone

        elif self.backbone == 'vgg11_bn':
            self.back = models.vgg11_bn()
            self.back = nn.Sequential(*list(self.back.children())[:-1][0])
            self.out_features = 512# None # TODO

        elif self.backbone == 'mobilenet_v3_small':
            self.back = models.mobilenet_v3_small()
            self.back = nn.Sequential(*list(self.back.children())[:-1][0])
            self.out_features = 576 #None # TODO

    def _forward(self, x):
        # TODO: forward pass through the backbone
        if self.backbone == 'resnet18':
            x= self.back(x)

        elif self.backbone == 'vgg11_bn':
            x= self.back(x)

        elif self.backbone == 'mobilenet_v3_small':
            x= self.back(x)

        return x

    def forward(self, inputs):
        #pass # TODO
        height,width = inputs.shape[2],inputs.shape[3]


        logits = self.head(self.aspp(self.back(inputs)))
        logits = nn.functional.interpolate(logits, size= (height,width), mode = 'bilinear', align_corners=False)

        assert logits.shape == (inputs.shape[0], self.num_classes, inputs.shape[2], inputs.shape[3]), 'Wrong shape of the logits'
        return logits


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_classes, 1)
        )


class ASPP(nn.Module):
    """
    TODO: 8 points

    Atrous Spatial Pyramid Pooling module
    with given atrous_rates and out_channels for each head
    Description: https://paperswithcode.com/method/aspp
    
    Detailed scheme: materials/deeplabv3.png
      - "Rates" are defined by atrous_rates
      - "Conv" denotes a Conv-BN-ReLU block
      - "Image pooling" denotes a global average pooling, followed by a 1x1 "conv" block and bilinear upsampling
      - The last layer of ASPP block should be Dropout with p = 0.5

    Args:
      - in_channels: number of input and output channels
      - num_channels: number of output channels in each intermediate "conv" block
      - atrous_rates: a list with dilation values
    """
    def __init__(self, in_channels, num_channels, atrous_rates):
        super(ASPP, self).__init__()
        self.paralels = nn.ModuleList([
                                       BlockConv2d(in_channels, num_channels, 1)]
                                      ) 

        for rate in atrous_rates:
            self.paralels.append(AtrousConv2d(in_channels,num_channels,3,rate))

        self.paralels.append(GlobalPool(in_channels, num_channels))
        #WHY out is in_channels ???????????????????????
        self.bottleneck = BlockConv2d(len(self.paralels)*num_channels, in_channels, 1 )

    def forward(self, x):
        # TODO: forward pass through the ASPP module
        res = []

        for module in self.paralels:
            res.append(module(x))
        
        res = torch.cat(res,dim=1)

        res = self.bottleneck(res)

        assert res.shape[1] == x.shape[1], 'Wrong number of output channels'
        assert res.shape[2] == x.shape[2] and res.shape[3] == x.shape[3], 'Wrong spatial size'
        
        return res