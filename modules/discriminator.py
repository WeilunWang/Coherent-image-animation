from torch import nn
import torch.nn.functional as F
from modules.util import kp2gaussian
from modules.util import zip_dimT_to_dimBS, unzip_dimT_from_dimBS
import torch


class DownBlock2d(nn.Module):
    """
    Simple block for processing video (encoder).
    """

    def __init__(self, in_features, out_features, norm=False, kernel_size=4, pool=False, sn=False):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size)

        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)

        if norm:
            self.norm = nn.InstanceNorm2d(out_features, affine=True)
        else:
            self.norm = None
        self.pool = pool

    def forward(self, x):
        out = x
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        out = F.leaky_relu(out, 0.2)
        if self.pool:
            out = F.avg_pool2d(out, (2, 2))
        return out


class DownBlock3d(nn.Module):
    """
    Simple block for processing video (encoder).
    """

    def __init__(self, in_features, out_features, norm=False, kernel_size=(3, 4, 4), pool=False, sn=False):
        super(DownBlock3d, self).__init__()
        self.pad = nn.ReplicationPad3d((0, 0, 0, 0, 1, 1))
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding=0)

        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)

        if norm:
            self.norm = nn.InstanceNorm3d(out_features, affine=True)
        else:
            self.norm = None
        self.pool = pool

    def forward(self, x):
        out = x
        out = self.conv(self.pad(out))
        if self.norm:
            out = self.norm(out)
        out = F.leaky_relu(out, 0.2, inplace=True)
        if self.pool:
            out = F.avg_pool3d(out, (1, 2, 2))
        return out


class ImageDiscriminator(nn.Module):
    """
    Discriminator similar to Pix2Pix
    """

    def __init__(self, num_channels=3, block_expansion=64, num_blocks=4, max_features=512,
                 sn=False, use_kp=False, num_kp=10, kp_variance=0.01, **kwargs):
        super(ImageDiscriminator, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(
                DownBlock2d(num_channels + num_kp * use_kp if i == 0 else min(max_features, block_expansion * (2 ** i)),
                            min(max_features, block_expansion * (2 ** (i + 1))),
                            norm=(i != 0), kernel_size=4, pool=(i != num_blocks - 1), sn=sn))

        self.down_blocks = nn.ModuleList(down_blocks)
        self.conv = nn.Conv2d(self.down_blocks[-1].conv.out_channels, out_channels=1, kernel_size=1)
        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)
        self.use_kp = use_kp
        self.kp_variance = kp_variance

    def forward(self, x, kp=None):
        feature_maps = []
        out = x
        if self.use_kp:
            shape = kp['value'].shape
            value = kp['value'].permute(0, 2, 1, 3).contiguous().view(-1, shape[1], shape[3])
            kp = {'value': value}
            heatmap = kp2gaussian(kp, x.shape[3:], self.kp_variance)
            heatmap = heatmap.view(shape[0], shape[2], shape[1], heatmap.shape[2], heatmap.shape[3])
            heatmap = heatmap.contiguous().permute(0, 2, 1, 3, 4)
            out = torch.cat([out, heatmap], dim=1)

        out = zip_dimT_to_dimBS(out)
        for down_block in self.down_blocks:
            feature_maps.append(down_block(out))
            out = feature_maps[-1]
        prediction_map = self.conv(out)

        return feature_maps, prediction_map


class VideoDiscriminator(nn.Module):
    """
    Discriminator similar to Pix2Pix
    """

    def __init__(self, num_channels=3, num_frame=4, block_expansion=64, num_blocks=4, max_features=512,
                 sn=False, use_kp=False, num_kp=10, kp_variance=0.01, **kwargs):
        super(VideoDiscriminator, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(
                DownBlock3d(num_channels + num_kp * use_kp if i == 0 else min(max_features, block_expansion * (2 ** i)),
                            min(max_features, block_expansion * (2 ** (i + 1))),
                            norm=(i != 0), kernel_size=(3, 4, 4), pool=(i != num_blocks - 1), sn=sn))

        self.down_blocks = nn.ModuleList(down_blocks)
        self.conv = nn.Conv3d(self.down_blocks[-1].conv.out_channels, out_channels=1, kernel_size=1)
        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)
        self.use_kp = use_kp
        self.kp_variance = kp_variance

    def forward(self, x, kp=None):
        feature_maps = []
        out = x
        if self.use_kp:
            shape = kp['value'].shape
            value = kp['value'].permute(0, 2, 1, 3).contiguous().view(-1, shape[1], shape[3])
            kp = {'value': value}
            heatmap = kp2gaussian(kp, x.shape[3:], self.kp_variance)
            heatmap = heatmap.view(shape[0], shape[2], shape[1], heatmap.shape[2], heatmap.shape[3])
            heatmap = heatmap.contiguous().permute(0, 2, 1, 3, 4)
            out = torch.cat([out, heatmap], dim=1)

        for down_block in self.down_blocks:
            feature_maps.append(down_block(out))
            out = feature_maps[-1]
        prediction_map = self.conv(out)

        return feature_maps, prediction_map


class MultiScaleImageDiscriminator(nn.Module):
    """
    Multi-scale (scale) discriminator
    """

    def __init__(self, scales=(), **kwargs):
        super(MultiScaleImageDiscriminator, self).__init__()
        self.scales = scales
        discs = {}
        for scale in scales:
            discs[str(scale).replace('.', '-')] = ImageDiscriminator(**kwargs)
        self.discs = nn.ModuleDict(discs)

    def forward(self, x, kp=None):
        out_dict = {}
        for scale, disc in self.discs.items():
            scale = str(scale).replace('-', '.')
            key = 'prediction_' + scale
            feature_maps, prediction_map = disc(x[key], kp)
            out_dict['feature_maps_' + scale] = feature_maps
            out_dict['prediction_map_' + scale] = prediction_map
        return out_dict


class MultiScaleVideoDiscriminator(nn.Module):
    """
    Multi-scale (scale) discriminator
    """

    def __init__(self, scales=(), **kwargs):
        super(MultiScaleVideoDiscriminator, self).__init__()
        self.scales = scales
        discs = {}
        for scale in scales:
            discs[str(scale).replace('.', '-')] = VideoDiscriminator(**kwargs)
        self.discs = nn.ModuleDict(discs)

    def forward(self, x, kp=None):
        out_dict = {}
        for scale, disc in self.discs.items():
            scale = str(scale).replace('-', '.')
            key = 'prediction_' + scale
            feature_maps, prediction_map = disc(x[key], kp)
            out_dict['feature_maps_' + scale] = feature_maps
            out_dict['prediction_map_' + scale] = prediction_map
        return out_dict
