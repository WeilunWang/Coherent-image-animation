from torch import nn
import torch
import torch.nn.functional as F
from modules.util import AntiAliasInterpolation2d, make_coordinate_grid
from modules.util import zip_dimT_to_dimBS, unzip_dimT_from_dimBS
from modules.util import SoftCrossEntropyLoss, MatrixEqualityLoss
from torchvision import models
import numpy as np
from torch.autograd import grad


class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class Pyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss.
    """
    def __init__(self, scales, num_channels):
        super(Pyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        shape = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(shape[0] * shape[2], shape[1], shape[3], shape[4])
        for scale, down_module in self.downs.items():
            s = float(str(scale).replace('-', '.'))
            out = down_module(x)
            out = out.view(shape[0], shape[2], shape[1], int(shape[3] * s),
                           int(shape[4] * s)).contiguous().permute(0, 2, 1, 3, 4)
            out_dict['prediction_' + str(scale).replace('-', '.')] = out
        return out_dict


class Transform:
    """
    Random tps transformation for equivariance constraints.
    """
    def __init__(self, bs, **kwargs):
        noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid((kwargs['points_tps'], kwargs['points_tps']), type=noise.type())
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0,
                                               std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 2]))
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, padding_mode="reflection")

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = torch.abs(distances).sum(-1)

            result = distances ** 2
            result = result * torch.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result

        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian


def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}


class GeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, kp_tracker, generator, image_discriminator, video_discriminator, train_params):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.kp_tracker = kp_tracker
        self.generator = generator
        self.image_discriminator = image_discriminator
        self.video_discriminator = video_discriminator
        self.train_params = train_params
        self.scales = train_params['scales']
        self.disc_scales_img = self.image_discriminator.scales
        self.disc_scales_vid = self.video_discriminator.scales
        self.pyramid = Pyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()

    def forward(self, x):
        kp_source = self.kp_extractor(x['source'])
        kp_driving_0 = self.kp_extractor(x['driving'][:, :, 0])
        kp_driving, kp_driving_inv, jacobian_t = self.kp_tracker(x['driving'], kp=kp_driving_0)

        kp_source_viz = {
            'value': kp_source['value'].unsqueeze(dim=2).repeat(1, 1, x['driving'].shape[2], 1),
            'jacobian': kp_source['jacobian'].unsqueeze(dim=2).repeat(1, 1, x['driving'].shape[2], 1, 1),
        }

        generated = self.generator(x['source'], kp_source=kp_source, kp_driving=kp_driving)
        generated.update({'kp_source': kp_source_viz, 'kp_driving': kp_driving})

        loss_values = {}

        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'])

        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(zip_dimT_to_dimBS(pyramide_generated['prediction_' + str(scale)]))
                y_vgg = self.vgg(zip_dimT_to_dimBS(pyramide_real['prediction_' + str(scale)]))

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
                loss_values['perceptual'] = value_total

        if self.loss_weights['generator_gan_img'] != 0:
            discriminator_maps_generated = self.image_discriminator(pyramide_generated, kp=detach_kp(kp_driving))
            discriminator_maps_real = self.image_discriminator(pyramide_real, kp=detach_kp(kp_driving))
            value_total = 0
            for scale in self.disc_scales_img:
                key = 'prediction_map_%s' % scale
                value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                value_total += self.loss_weights['generator_gan_img'] * value
            loss_values['gen_gan_img'] = value_total

            if sum(self.loss_weights['feature_matching_img']) != 0:
                value_total = 0
                for scale in self.disc_scales_img:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching_img'][i] == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching_img'][i] * value
                    loss_values['feature_matching_img'] = value_total

        if self.loss_weights['generator_gan_vid'] != 0:
            discriminator_maps_generated = self.video_discriminator(pyramide_generated, kp=detach_kp(kp_driving))
            discriminator_maps_real = self.video_discriminator(pyramide_real, kp=detach_kp(kp_driving))
            value_total = 0
            for scale in self.disc_scales_vid:
                key = 'prediction_map_%s' % scale
                value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                value_total += self.loss_weights['generator_gan_vid'] * value
            loss_values['gen_gan_vid'] = value_total

            if sum(self.loss_weights['feature_matching_vid']) != 0:
                value_total = 0
                for scale in self.disc_scales_vid:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching_vid'][i] == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching_vid'][i] * value
                    loss_values['feature_matching_vid'] = value_total

        if (self.loss_weights['equivariance_value'] + self.loss_weights['equivariance_jacobian']) != 0:
            transform = Transform(x['driving'].shape[0], **self.train_params['transform_params'])
            transformed_frame = transform.transform_frame(x['driving'][:, :, 0])
            transformed_kp = self.kp_extractor(transformed_frame)

            generated['transformed_frame'] = transformed_frame
            generated['transformed_kp'] = transformed_kp

            ## Value loss part
            if self.loss_weights['equivariance_value'] != 0:
                value = torch.abs(kp_driving['value'][:, :, 0] - transform.warp_coordinates(transformed_kp['value'])).mean()
                loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value

            ## jacobian loss part
            if self.loss_weights['equivariance_jacobian'] != 0:
                jacobian_transformed = torch.matmul(transform.jacobian(transformed_kp['value']),
                                                    transformed_kp['jacobian'])

                normed_source = torch.inverse(kp_driving['jacobian'][:, :, 0])
                normed_transformed = jacobian_transformed
                value = torch.matmul(normed_source, normed_transformed)

                eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())

                value = torch.abs(eye - value).mean()
                loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * value

        if sum(self.loss_weights['cycle_consistency_loss']) != 0:
            value_total = 0

            # prediction_value_cycle(cross entropy loss)
            final_shape = kp_driving['prediction_value'].shape
            heatmap = kp_driving['prediction_value'].view(-1, final_shape[3] * final_shape[4])
            heatmap_inv = kp_driving_inv['prediction_value'].view(-1, final_shape[3] * final_shape[4])
            value = SoftCrossEntropyLoss(heatmap_inv, heatmap)
            value_total += self.loss_weights['cycle_consistency_loss'][0] * value

            # prediciton_jacobian_cycle
            value = MatrixEqualityLoss(kp_driving_inv['jacobian'], kp_driving['jacobian'])
            value_total += self.loss_weights['cycle_consistency_loss'][1] * value
            loss_values['cycle_consistency_loss'] = value_total

        if sum(self.loss_weights['adjacent_matrix_regularization']) != 0:
            value_total = 0

            # adjacent_matrix_regularization
            num_jacobian_maps = jacobian_t.shape[1] // 4
            eye = torch.tensor([1, 0, 0, 1] * num_jacobian_maps, dtype=jacobian_t.dtype)
            if torch.cuda.is_available():
                eye = eye.cuda()
            eye = eye.view(1, -1, 1, 1, 1)
            value = torch.abs(jacobian_t - eye).mean()
            value_total += self.loss_weights['adjacent_matrix_regularization'] * value

            loss_values['adjacent_matrix_regularization'] = value_total

        return loss_values, generated


class DiscriminatorFullModel(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, kp_tracker, generator, image_discriminator, video_discriminator, train_params):
        super(DiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.kp_tracker = kp_tracker
        self.generator = generator
        self.image_discriminator = image_discriminator
        self.video_discriminator = video_discriminator
        self.train_params = train_params
        self.scales_img = self.image_discriminator.scales
        self.scales_vid = self.video_discriminator.scales
        self.pyramid = Pyramide(self.scales_img, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

    def forward(self, x, generated):
        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'].detach())

        kp_driving = generated['kp_driving']
        discriminator_maps_generated_img = self.image_discriminator(pyramide_generated, kp=detach_kp(kp_driving))
        discriminator_maps_real_img = self.image_discriminator(pyramide_real, kp=detach_kp(kp_driving))

        discriminator_maps_generated_vid = self.video_discriminator(pyramide_generated, kp=detach_kp(kp_driving))
        discriminator_maps_real_vid = self.video_discriminator(pyramide_real, kp=detach_kp(kp_driving))

        loss_values = {}
        value_total = 0
        for scale in self.scales_img:
            key = 'prediction_map_%s' % scale
            value = (1 - discriminator_maps_real_img[key]) ** 2 + discriminator_maps_generated_img[key] ** 2
            value_total += self.loss_weights['discriminator_gan_img'] * value.mean()
        loss_values['disc_gan_img'] = value_total

        value_total = 0
        for scale in self.scales_vid:
            key = 'prediction_map_%s' % scale
            value = (1 - discriminator_maps_real_vid[key]) ** 2 + discriminator_maps_generated_vid[key] ** 2
            value_total += self.loss_weights['discriminator_gan_vid'] * value.mean()
        loss_values['disc_gan_vid'] = value_total

        return loss_values

