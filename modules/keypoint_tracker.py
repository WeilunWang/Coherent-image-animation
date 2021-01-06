from torch import nn
import torch
import torch.nn.functional as F
from correlation_package import Correlation
from modules.util import make_coordinate_grid, AntiAliasInterpolation2d
from modules.util import DownBlock2d, UpBlock2d
from modules.util import zip_dimT_to_dimBS, unzip_dimT_from_dimBS


class FeatureExtractor(nn.Module):
    """
    Extracting image feature.
    """

    def __init__(self, block_expansion, in_features, out_features, max_features, num_blocks):
        super(FeatureExtractor, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_blocks)[::-1]:
            in_filters = (out_features if i != num_blocks - 1 else 0) + min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = out_features
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = out_features

    def forward(self, x):
        down_outs = [x]
        for i in range(len(self.down_blocks)):
            down_outs.append(self.down_blocks[i](down_outs[-1]))
        out = down_outs.pop()
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
            skip = down_outs.pop()
            if i != len(self.up_blocks) - 1:
                out = torch.cat([out, skip], dim=1)
        return out


class TemporalCorrespondanceModule(nn.Module):
    """
    Extracting video motion. Return motion info.
    """

    def __init__(self, block_expansion, num_kp, num_channels, out_features, max_features, max_displacement,
                 num_blocks, temperature, temperature_kp, estimate_jacobian=False, scale_factor=1):
        super(TemporalCorrespondanceModule, self).__init__()

        self.predictor = FeatureExtractor(block_expansion, in_features=num_channels, out_features=out_features,
                                          max_features=max_features, num_blocks=num_blocks)

        self.corr = Correlation(pad_size=max_displacement, kernel_size=1, max_displacement=max_displacement,
                                stride1=1, stride2=1, corr_multiply=1)
        self.pad = nn.ReplicationPad2d(padding=max_displacement)

        if estimate_jacobian:
            self.num_jacobian_maps = num_kp
            self.jacobian = nn.Conv2d(in_channels=self.corr.out_channel,
                                      out_channels=4 * self.num_jacobian_maps, kernel_size=(7, 7), padding=3)
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))
        else:
            self.jacobian = None

        self.temperature = temperature
        self.temperature_kp = temperature_kp
        self.max_displacement = max_displacement
        self.scale_factor = scale_factor
        self.estimate_jacobian = estimate_jacobian
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    def gaussian2kp(self, heatmap):
        """
        Extract the mean and from a heatmap
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
        value = (heatmap * grid).sum(dim=(2, 3))
        kp = {'value': value}

        return kp

    def forward(self, frame0, frame1, kp0):
        if self.scale_factor != 1:
            frame0 = self.down(frame0)
            frame1 = self.down(frame1)

        # figure out correlation between frame 0 & 1
        feature_map0 = self.predictor(frame0)
        feature_map0 = feature_map0 - feature_map0.mean(dim=(2, 3), keepdim=True)
        feature_map0 = feature_map0 / torch.sum(feature_map0 ** 2, dim=1).sqrt().unsqueeze(dim=1)

        feature_map1 = self.predictor(frame1)
        feature_map1 = feature_map1 - feature_map1.mean(dim=(2, 3), keepdim=True)
        feature_map1 = feature_map1 / torch.sum(feature_map1 ** 2, dim=1).sqrt().unsqueeze(dim=1)

        correlation = self.corr(feature_map1, feature_map0)
        probability = F.softmax(correlation / self.temperature, dim=1)
        probability = probability.unsqueeze(dim=2)

        # figure out new heatmap
        final_shape = kp0['prediction_value'].shape
        prediction_value0 = kp0['prediction_value']
        prediction_value0_pad = self.pad(prediction_value0)
        prediction_value0_neighbourhood = torch.stack(
            [prediction_value0_pad[:, :, i:i + final_shape[2], j:j + final_shape[3]] for i in
             range(2 * self.max_displacement + 1) for j in range(2 * self.max_displacement + 1)], dim=1)
        prediction_value1 = torch.sum(probability * prediction_value0_neighbourhood, dim=1)
        heatmap1 = prediction_value1.view(final_shape[0], final_shape[1], -1)
        heatmap1 = F.softmax(heatmap1 / self.temperature_kp, dim=2)
        heatmap1 = heatmap1.view(*final_shape)

        kp1 = self.gaussian2kp(heatmap1)
        kp1['prediction_value'] = prediction_value1

        jacobian_map_temporal = None
        if self.estimate_jacobian:
            jacobian_map0 = kp0['prediction_jacobian']

            jacobian_map0_pad = self.pad(jacobian_map0)
            jacobian_map0_neighbourhood = torch.stack(
                [jacobian_map0_pad[:, :, i:i + final_shape[2], j:j + final_shape[3]] for i in
                 range(2 * self.max_displacement + 1) for j in range(2 * self.max_displacement + 1)], dim=1)
            jacobian_map0_1 = torch.sum(probability * jacobian_map0_neighbourhood, dim=1)
            jacobian_map0_1 = jacobian_map0_1.view(final_shape[0], self.num_jacobian_maps, 4,
                                                   final_shape[2], final_shape[3]).contiguous()
            jacobian_map0_1 = jacobian_map0_1.permute(0, 1, 3, 4, 2).contiguous().view(-1, 2, 2)

            jacobian_map = self.jacobian(correlation)
            jacobian_map_temporal = jacobian_map
            jacobian_map = jacobian_map.reshape(final_shape[0], self.num_jacobian_maps, 4, final_shape[2],
                                                final_shape[3])
            jacobian_map = jacobian_map.permute(0, 1, 3, 4, 2).contiguous().view(-1, 2, 2)
            jacobian_map1 = torch.bmm(jacobian_map, jacobian_map0_1)
            jacobian_map1 = jacobian_map1.view(final_shape[0], self.num_jacobian_maps, final_shape[2],
                                               final_shape[3], 4).contiguous().permute(0, 1, 4, 2, 3).contiguous()
            kp1['prediction_jacobian'] = jacobian_map1.view(final_shape[0], -1, final_shape[2], final_shape[3])
            heatmap1 = heatmap1.unsqueeze(2)

            jacobian1 = heatmap1 * jacobian_map1
            jacobian1 = jacobian1.view(final_shape[0], final_shape[1], 4, -1)
            jacobian1 = jacobian1.sum(dim=-1)
            jacobian1 = jacobian1.view(jacobian1.shape[0], jacobian1.shape[1], 2, 2)
            kp1['jacobian'] = jacobian1

        return jacobian_map_temporal, kp1


class KPTracker(nn.Module):
    def __init__(self, num_channels, num_kp, estimate_jacobian=False, temporal_correspondance_params=None):
        super(KPTracker, self).__init__()

        self.temporal_correspondance = TemporalCorrespondanceModule(num_kp=num_kp, num_channels=num_channels,
                                                                    estimate_jacobian=estimate_jacobian,
                                                                    **temporal_correspondance_params)

    def forward(self, video, kp):
        num_frame = video.shape[2]

        kp_list = [kp]
        jacobian_map_temporal_list = []
        for i in range(num_frame - 1):
            jacobian_map_temporal, kp_new = self.temporal_correspondance(video[:, :, i], video[:, :, i + 1], kp_list[-1])
            kp_list.append(kp_new)
            jacobian_map_temporal_list.append(jacobian_map_temporal)
        jacobian_map_temporal_video = torch.stack(jacobian_map_temporal_list, dim=2)
        kp_video = dict()
        for key in kp.keys():
            kp_video[key] = torch.stack([kp[key] for kp in kp_list], dim=2)

        kp_list_inv = [kp_list[-1]]
        corr_list = []
        for i in range(num_frame - 1):
            _, kp_new = self.temporal_correspondance(video[:, :, num_frame-1-i], video[:, :, num_frame-i-2], kp_list_inv[-1])
            kp_list_inv.append(kp_new)

        kp_video_inv = dict()
        for key in kp.keys():
            kp_video_inv[key] = torch.stack([kp[key] for kp in kp_list_inv[::-1]], dim=2)

        return kp_video, kp_video_inv, jacobian_map_temporal_video


