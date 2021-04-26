import torch
from torch.nn import Conv2d, ReLU, BatchNorm2d, Dropout, ConvTranspose2d, MaxPool2d

class DownBlock(torch.nn.Module):
    def __init__(self, in_ch, mid_ch = None , out_ch = None):
        super(DownBlock, self).__init__()

        if out_ch == None: out_ch = 2*in_ch
        if mid_ch == None: mid_ch = out_ch

        self.conv1 = Conv2d(in_channels = in_ch , out_channels = mid_ch, kernel_size = 3, stride = 1, padding = 1)
        self.batn1 = BatchNorm2d(mid_ch) 
        self.act1  = ReLU()
        
        self.conv2 = Conv2d(in_channels = mid_ch , out_channels = out_ch , kernel_size = 3, stride = 1, padding = 1)
        self.batn2 = BatchNorm2d(out_ch)
        self.act2  = ReLU()
        
        self.pool  = MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.batn1(x) 
        x = self.act1(x)

        x = self.conv2(x)
        x = self.batn2(x)
        x = self.act2(x)

        skip = x

        x = self.pool(x)

        return x, skip

class BaseBlock(torch.nn.Module):
    def __init__(self, in_ch):
        super(BaseBlock, self).__init__()
        
        self.block = torch.nn.Sequential(
            Conv2d(in_channels=in_ch, out_channels = in_ch, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Conv2d(in_channels=in_ch, out_channels = in_ch, kernel_size=3, stride=1, padding=1),
            ReLU(),
            ConvTranspose2d(in_channels = in_ch, out_channels= in_ch, kernel_size = 3, padding = 1, output_padding=1 , stride = 2)
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(torch.nn.Module):
    def __init__(self, in_ch , mid_ch = None, out_ch = None, isUpconv = True):
        super(UpBlock, self).__init__()
        
        if out_ch == None: out_ch = in_ch*2 // 4
        if mid_ch == None: mid_ch = out_ch
        #self.skip = skip
        self.isUpconv = isUpconv

        self.drop1 = Dropout(p = 0.5)
        
        self.conv1 = Conv2d(in_channels = in_ch*2, out_channels = mid_ch, kernel_size = 3, stride= 1, padding=1)
        self.batn1 = BatchNorm2d(mid_ch)
        self.act1  = ReLU()
        
        self.conv2 = Conv2d(in_channels = mid_ch, out_channels = out_ch, kernel_size = 3, stride= 1, padding=1)
        self.batn2 = BatchNorm2d(out_ch)
        self.act2 = ReLU()
        
        if isUpconv:
            self.ups1  = ConvTranspose2d(in_channels = out_ch, out_channels= out_ch, kernel_size =3, padding =1, output_padding = 1, stride = 2)
        

    def forward(self, x, skip):

        x = torch.cat((skip, x), dim = 1)
        
        x = self.drop1(x)
        
        x = self.conv1(x)
        x = self.batn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.batn2(x)
        x = self.act2(x)

        if self.isUpconv:
            x = self.ups1(x)
        
        return x 

class Unet(torch.nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        self.down1 = DownBlock(1,56,112)
        self.down2 = DownBlock(112,224)
        self.down3 = DownBlock(224,448)
        self.base  = BaseBlock(448)
        self.up1   = UpBlock(448)
        self.up2   = UpBlock(224)
        self.up3   = UpBlock(112, 112, 56, False)

    def forward(self, x):
        
        x, skip1 = self.down1(x)
        #print('down1: ', x.shape) 
        x, skip2 = self.down2(x)
        #print('down2: ', x.shape)
        x,skip3 = self.down3(x)
        #print('down3: ', x.shape)
        x = self.base(x)
        #print('base: ', x.shape)
        x = self.up1(x,skip3)
        #print('up1: ', x.shape)
        x = self.up2(x,skip2)
        #print('up2: ', x.shape)
        x = self.up3(x,skip1)

        
        return x

