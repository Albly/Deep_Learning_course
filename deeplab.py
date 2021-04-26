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
            pass

        elif self.backbone == 'vgg11_bn':
            pass

        elif self.backbone == 'mobilenet_v3_small':
            pass

        return x

    def forward(self, inputs):
        #pass # TODO

        height, width = inputs.shape[2], inputs.shape[3]

        logits = self.head(self.aspp(self.back(inputs)))
        logits = nn.functional.interpolate(logits, size=(height,width), mode = 'bilinear', align_corners=False)

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
            self.paralels.append(AtrousConv2d(in_channels, num_channels,3,rate))
        
        self.paralels.append(GlobalPool(in_channels, num_channels))
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