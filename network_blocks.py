import torch
from torch import optim, cuda, nn

class Conv_Block(nn.Module):
    """(convolution => ReLU)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)

class Down(nn.Module):
    """Double Conv then downscale with Max Pool"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            Conv_Block(in_channels,out_channels),
            Conv_Block(out_channels,out_channels)
        )
        self.maxpool_conv = nn.MaxPool3d(2)
        
    def forward(self, x):
        x = self.conv_block(x)
        return x,self.maxpool_conv(x)
    
class Bottom(nn.Module):
    """Double Conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            Conv_Block(in_channels,out_channels),
            Conv_Block(out_channels,out_channels)
        )
        
    def forward(self, x):
        x = self.conv_block(x)
        return x


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        #self.up = nn.Upsample(scale_factor=2)#, mode='bilinear', align_corners=True)
        self.conv = self.conv_block = nn.Sequential(
            Conv_Block(in_channels,out_channels),
            Conv_Block(out_channels,out_channels)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class STN_Block(nn.Module):
    """(convolution => ReLU)"""
    def __init__(self, in_channels, out_ch1, out_ch2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_ch1, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_ch1, out_ch2, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.linear = nn.Sequential(
            nn.Linear(47250, 500),
            nn.ReLU(),
            nn.Linear(500, 12)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(1,-1)
        x = self.linear(x)
        return x
    
class SpatialTransform(nn.Module):
    def __init__(self):
        super(SpatialTransform, self).__init__()
        
        self.prepare = STN_Block(1,8,10)

    def forward(self, image_):
        
         # affine matrix 
        theta = self.prepare(image_)
        theta = theta.view(-1, 3, 4) # 3x4 affine matrix
        # displacement field 
        grid_shape = image_.size()
        displacement_grid = nn.functional.affine_grid(theta, grid_shape)
        # warp 
        deformed_image = nn.functional.grid_sample(image_, displacement_grid)
        
        return deformed_image
    
class FinalTrafo(nn.Module):
    def __init__(self):
        super(FinalTrafo, self).__init__()
        
        self.prepare = STN_Block(2,8,10)

    def forward(self, img, pred):
        
        theta = self.prepare(img)
        theta = theta.view(-1, 3, 4) # 3x4 affine matrix
        # displacement field 
        grid_shape = pred.size()
        displacement_grid = nn.functional.affine_grid(theta, grid_shape)
        # warp 
        deformed_image = nn.functional.grid_sample(pred, displacement_grid)
        
        return deformed_image
