import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock3d, UpBlock3d, SameBlock2d, DownBlock2d
from modules.util import zip_dimT_to_dimBS, unzip_dimT_from_dimBS
from modules.dense_motion import DenseMotionNetwork


class OcclusionAwareVideoGenerator(nn.Module):
    """
    Generator that given source image and and keypoints try to transform image according to movement trajectories
    induced by keypoints. Generator follows Johnson architecture.
    """

    def __init__(self, num_channels, num_kp, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False):
        super(OcclusionAwareVideoGenerator, self).__init__()

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, num_channels=num_channels,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None

        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock3d(in_features, out_features, kernel_size=(3, 3, 3), padding=(1, 1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock3d(in_features, kernel_size=(3, 3, 3), padding=(1, 1, 1)))

        self.final = nn.Conv3d(block_expansion, num_channels, kernel_size=(3, 7, 7), padding=(1, 3, 3))
        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear')
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation)

    def forward(self, source_image, kp_driving, kp_source):
        num_frame = kp_driving['value'].shape[2]

        # Encoding (downsampling) part
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        out = zip_dimT_to_dimBS(out.unsqueeze(dim=2).repeat(1, 1, num_frame, 1, 1))

        # Transforming feature representation according to deformation and occlusion
        output_dict = dict()
        source_image = zip_dimT_to_dimBS(source_image.unsqueeze(dim=2).repeat(1, 1, num_frame, 1, 1))
        kp_source = {'value': zip_dimT_to_dimBS(kp_source['value'].unsqueeze(dim=2).repeat(1, 1, num_frame, 1)),
                     'jacobian': zip_dimT_to_dimBS(kp_source['jacobian'].unsqueeze(dim=2).repeat(1, 1, num_frame, 1, 1))}
        kp_driving = {'value': zip_dimT_to_dimBS(kp_driving['value']),
                      'jacobian': zip_dimT_to_dimBS(kp_driving['jacobian'])}
        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(source_image=source_image, kp_driving=kp_driving,
                                                     kp_source=kp_source)
            output_dict['mask'] = unzip_dimT_from_dimBS(num_frame, dense_motion['mask'])
            output_dict['sparse_deformed'] = unzip_dimT_from_dimBS(num_frame, dense_motion['sparse_deformed'])

            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
                output_dict['occlusion_map'] = unzip_dimT_from_dimBS(num_frame, occlusion_map)
            else:
                occlusion_map = None
            deformation = dense_motion['deformation']
            out = self.deform_input(out, deformation)

            if occlusion_map is not None:
                if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear')
                out = out * occlusion_map

            output_dict["deformed"] = unzip_dimT_from_dimBS(num_frame, self.deform_input(source_image, deformation))

        out = unzip_dimT_from_dimBS(num_frame, out)

        # Decoding part
        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        out = F.sigmoid(out)

        output_dict["prediction"] = out

        return output_dict
