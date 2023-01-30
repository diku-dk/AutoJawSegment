import torch
import torchvision
import torchvision.transforms as transforms
import gzip
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, name=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding='same')
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding='same')

    def forward(self, x):
        x = self.relu(self.conv2(self.relu(self.conv1(x))))
        return self.bn(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, name=None):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch,
                                       kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.relu(self.conv(x))
        return self.bn(x)

class Encoder(nn.Module):
    def __init__(self, init_filters=64, n_channels=1, depth=4, cf=1):
        super().__init__()
        self.chs = [n_channels] + [int(init_filters * cf * 2**i) for i in range(depth)]
        self.enc_blocks = nn.ModuleList([Block(self.chs[i], self.chs[i + 1])
                                         for i in range(len(self.chs) - 1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return x, ftrs

class Bottom(nn.Module):
    def __init__(self, chs_in=1024, chs_out=1024):
        super().__init__()
        self.bottom_blocks = Block(chs_in, chs_out)

    def forward(self, x):
        x = self.bottom_blocks(x)
        return x



class Decoder(nn.Module):
    def __init__(self, init_filters=64, depth=4, cf=1, n_classes=1, flatten_output=False):
        super().__init__()
        chs = [int(init_filters * cf * 2**(depth-i)) for i in range(depth)] + [int(init_filters * cf)]
        self.n_classes = n_classes
        self.chs = chs
        self.upconvs = nn.ModuleList([UpBlock(self.chs[i], self.chs[i + 1])
                                      for i in range(len(self.chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(self.chs[i], self.chs[i + 1])
                                         for i in range(len(self.chs) - 1)])

        self.final_conv = nn.Conv2d(self.chs[-1], n_classes, 1)
        self.flatten_output = flatten_output

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)

        x = self.final_conv(x)
        # note: do not do softmax or sigmoid since it is handled by loss func

        # out = F.interpolate(out, self.out_sz)

        if self.flatten_output:
            if self.n_classes != 1:
                x = x.contiguous().view(x.size(0), -1, self.n_classes)
            else:
                x = x.contiguous().view(x.size(0), -1)

        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(nn.Module):
    def __init__(self,
                 n_classes,
                 img_rows=None,
                 img_cols=None,
                 dim=None,
                 n_channels=1,
                 depth=4,
                 out_activation="softmax",
                 activation="relu",
                 kernel_size=3,
                 padding="same",
                 complexity_factor=1,
                 flatten_output=False,
                 l2_reg=None,
                 logger=None,
                 build_res=False,
                 init_filters=64,
                 weight_map=False,):
        super().__init__()
        self.cf = np.sqrt(complexity_factor)
        self.img_shape = (n_channels, img_rows, img_cols)
        self.flatten_output = flatten_output
        self.n_channels = n_channels
        self.init_filters = init_filters
        self.depth = depth
        self.n_classes = n_classes
        self.encoder = Encoder(init_filters=self.init_filters,
                               n_channels=self.n_channels, depth=self.depth, cf=self.cf)

        final_ch = int(self.init_filters * self.cf * 2**self.depth)//2

        self.bottom = Bottom(chs_in=final_ch, chs_out=final_ch*2)
        self.decoder = Decoder(init_filters=self.init_filters, n_classes=self.n_classes,
                             depth=self.depth, cf=self.cf, flatten_output=self.flatten_output)

        self.n_classes = n_classes

    def forward(self, x):
        x, enc_ftrs = self.encoder(x)

        x = self.bottom(x)

        out = self.decoder(x, enc_ftrs[::-1])

        return out

    def log(self):
        self.logger("UNet Model Summary\n------------------")
        self.logger("Image rows:        %i" % self.img_shape[0])
        self.logger("Image cols:        %i" % self.img_shape[1])
        self.logger("Image channels:    %i" % self.img_shape[2])
        self.logger("N classes:         %i" % self.n_classes)
        self.logger("CF factor:         %.3f" % self.cf ** 2)
        self.logger("Depth:             %i" % self.depth)
        self.logger("l2 reg:            %s" % self.l2_reg)
        self.logger("Padding:           %s" % self.padding)
        self.logger("Conv activation:   %s" % self.activation)
        self.logger("Out activation:    %s" % self.out_activation)
        self.logger("Receptive field:   %s" % self.receptive_field)
        self.logger("N params:          %i" % self.count_params())
        self.logger("Output:            %s" % self.output)
        self.logger("Crop:              %s" % (self.label_crop if np.sum(self.label_crop) != 0 else "None"))

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet(n_classes=3)
    model = model.to(device)
    img = torch.rand(2, 1, 64, 64)
    img = img.to(device)
    res = model(img)