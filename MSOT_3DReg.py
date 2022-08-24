import torch
from torch import optim, cuda, nn

import network_blocks as nb

class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        
        self.ST1 = nb.SpatialTransform()
        self.final = nb.FinalTrafo()

    def forward(self, pred, target):
        
        pretransform = self.ST1(pred)
        image = torch.cat([pretransform, target], dim=1)
        deformed_image = self.final(image, pred)
        
        return deformed_image

class Seg_UNet(nn.Module):
    def __init__(self, bilinear=False):
        super(Seg_UNet, self).__init__()
        
        self.initial = nb.Down(1,64)
        self.down1 = nb.Down(64, 128)
        self.down2 = nb.Down(128, 256)
        self.down3 = nb.Down(256, 512)
        self.floor = nb.Bottom(512,1024)
        self.up1 = nb.Up(1024, 512)
        self.up2 = nb.Up(512, 256)
        self.up3 = nb.Up(256, 128)
        self.up4 = nb.Up(128, 64)
        self.out = nb.Conv_Block(64,1)

    def forward(self, x):
        x_c1, x1 = self.initial(x) 
        x_c2, x2 = self.down1(x1)
        x_c3, x3 = self.down2(x2)
        x_c4, x4 = self.down3(x3)
        x5 = self.floor(x4)
        x = self.up1(x5, x_c4)
        x = self.up2(x, x_c3)
        x = self.up3(x, x_c2)
        x = self.up4(x, x_c1)
        return self.out(x)
      
class MSOT_3DReg(nn.Module):
    """
    Joint model for segmentation and registration
    """
    def __init__(self,
                 inshape
                 ):
        
        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        
        # configure core unet model
        self.unet_model = Seg_UNet()
        
        # configure transformer
        self.transformer = STN()

    def forward(self, source, target, registration=True):
        '''
        Parameters:
            source: Source image tensor (MSOT).
            target: Target image tensor (MRI).
            registration: Return transformed image and flow. Default is False.
        '''
        x_seg = self.unet_model(source)
        x = self.transformer(x_seg, target)
        
        return x, x_seg
