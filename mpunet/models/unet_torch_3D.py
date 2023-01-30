from unet_torch import *


class Block3D(Block):
    def __init__(self, in_ch, out_ch, name=None):
        super().__init__(in_ch, out_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding='same')
        self.bn = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding='same')


class UpBlock3D(UpBlock):
    def __init__(self, in_ch, out_ch):
        super().__init__(in_ch=in_ch, out_ch=out_ch)
        self.bn = nn.BatchNorm3d(out_ch)
        self.conv = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)


class Encoder3D(Encoder):
    def __init__(self, init_filters=64, n_channels=1, depth=4, cf=1):
        super().__init__(init_filters=init_filters, n_channels=n_channels, depth=depth, cf=cf)
        self.enc_blocks = nn.ModuleList([Block3D(self.chs[i], self.chs[i + 1])
                                         for i in range(len(self.chs) - 1)])
        self.pool = nn.MaxPool3d(2)

class Bottom3D(Bottom):
    def __init__(self, chs_in=1024, chs_out=1024):
        super().__init__(chs_in=chs_in, chs_out=chs_out)
        self.bottom_blocks = Block3D(chs_in, chs_out)


class Decoder3D(Decoder):
    def __init__(self, init_filters=64, depth=4, cf=1, n_classes=1, flatten_output=False):
        super().__init__(init_filters=init_filters, depth=depth, cf=cf,
                         n_classes=n_classes, flatten_output=flatten_output)

        self.upconvs = nn.ModuleList([UpBlock3D(self.chs[i], self.chs[i + 1])
                                      for i in range(len(self.chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block3D(self.chs[i], self.chs[i + 1])
                                         for i in range(len(self.chs) - 1)])
        self.final_conv = nn.Conv3d(self.chs[-1], n_classes, 1)


    def crop(self, enc_ftrs, x):
        _, _, H, W, D  = x.shape
        if not np.all(enc_ftrs.shape[-3:] == x.shape[-3:]):
            enc_ftrs = F.interpolate(enc_ftrs, [H, W, D])
        return enc_ftrs

        _, _, H, W, D = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W, D])(enc_ftrs)
        return enc_ftrs

class UNet3D(UNet):
    def __init__(self,
                 n_classes,
                 **kwargs):
        super().__init__(n_classes,**kwargs)
        # self.img_shape = (n_channels, img_rows, img_cols)

        self.encoder = Encoder3D(init_filters=self.init_filters ,
                                 n_channels=self.n_channels, depth=self.depth, cf=self.cf)

        final_ch = int(self.init_filters * self.cf * 2**self.depth)//2

        self.bottom = Bottom3D(chs_in=final_ch, chs_out=final_ch*2)
        self.decoder = Decoder3D(init_filters=self.init_filters, n_classes=self.n_classes,
                             depth=self.depth, cf=self.cf, flatten_output=self.flatten_output)


    def log(self):
        self.logger("UNet Model Summary\n------------------")
        self.logger("Image rows:        %i" % self.img_shape[0])
        self.logger("Image cols:        %i" % self.img_shape[1])
        self.logger("Image channels:    %i" % self.img_shape[2])

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet3D(n_classes=3, depth=3)
    model = model.to(device)
    img = torch.rand(2, 1, 64, 64, 64)
    img = img.to(device)
    res = model(img)