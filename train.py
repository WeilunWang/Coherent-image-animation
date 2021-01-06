from tqdm import trange
import torch

from torch.utils.data import DataLoader

from logger import Logger
from modules.model import GeneratorFullModel, DiscriminatorFullModel

from sync_batchnorm import DataParallelWithCallback

from frames_dataset import DatasetRepeater


def train(config, generator,  image_discriminator, video_discriminator, kp_detector,
          kp_tracker, checkpoint, log_dir, dataset, device_ids):
    train_params = config['train_params']

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    optimizer_image_discriminator = torch.optim.Adam(image_discriminator.parameters(), lr=train_params['lr_discriminator_img'], betas=(0.5, 0.999))
    optimizer_video_discriminator = torch.optim.Adam(video_discriminator.parameters(), lr=train_params['lr_discriminator_vid'], betas=(0.5, 0.999))
    optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))
    optimizer_kp_tracker = torch.optim.Adam(kp_tracker.parameters(), lr=train_params['lr_kp_tracker'], betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, generator, image_discriminator, video_discriminator, kp_detector, kp_tracker,
                                      optimizer_generator, optimizer_image_discriminator, optimizer_video_discriminator,
                                      None if train_params['lr_kp_detector'] == 0 else optimizer_kp_detector,
                                      None if train_params['lr_kp_tracker'] == 0 else optimizer_kp_tracker)

        # reset learning rate
        for param_group in optimizer_generator.param_groups:
            param_group['lr'] = train_params['lr_generator']
        for param_group in optimizer_image_discriminator.param_groups:
            param_group['lr'] = train_params['lr_discriminator_img']
        for param_group in optimizer_video_discriminator.param_groups:
            param_group['lr'] = train_params['lr_discriminator_vid']
        for param_group in optimizer_kp_detector.param_groups:
            param_group['lr'] = train_params['lr_kp_detector']
        for param_group in optimizer_kp_tracker.param_groups:
            param_group['lr'] = train_params['lr_kp_tracker']
    else:
        start_epoch = 0

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=6, drop_last=True)

    generator_full = GeneratorFullModel(kp_detector, kp_tracker, generator, image_discriminator,
                                        video_discriminator, train_params)
    discriminator_full = DiscriminatorFullModel(kp_detector, kp_tracker, generator, image_discriminator,
                                                video_discriminator, train_params)

    if torch.cuda.is_available():
        generator_full = DataParallelWithCallback(module=generator_full, device_ids=device_ids)
        discriminator_full = DataParallelWithCallback(module=discriminator_full, device_ids=device_ids)

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            for x in dataloader:

                losses_generator, generated = generator_full(x)

                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)

                loss.backward()
                optimizer_generator.step()
                optimizer_generator.zero_grad()
                optimizer_kp_detector.step()
                optimizer_kp_detector.zero_grad()
                optimizer_kp_tracker.step()
                optimizer_kp_tracker.zero_grad()

                if train_params['loss_weights']['generator_gan_img'] + train_params['loss_weights']['generator_gan_vid'] != 0:
                    optimizer_image_discriminator.zero_grad()
                    optimizer_video_discriminator.zero_grad()
                    losses_discriminator = discriminator_full(x, generated)
                    loss_values = [val.mean() for val in losses_discriminator.values()]
                    loss = sum(loss_values)

                    loss.backward()
                    optimizer_image_discriminator.step()
                    optimizer_image_discriminator.zero_grad()
                    optimizer_video_discriminator.step()
                    optimizer_video_discriminator.zero_grad()
                else:
                    losses_discriminator = {}

                losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                logger.log_iter(losses=losses)
            
            logger.log_epoch(epoch, {'generator': generator,
                                     'image_discriminator': image_discriminator,
                                     'video_discriminator': video_discriminator,
                                     'kp_detector': kp_detector,
                                     'kp_tracker': kp_tracker,
                                     'optimizer_generator': optimizer_generator,
                                     'optimizer_image_discriminator': optimizer_image_discriminator,
                                     'optimizer_video_discriminator': optimizer_video_discriminator,
                                     'optimizer_kp_detector': optimizer_kp_detector,
                                     'optimizer_kp_tracker': optimizer_kp_tracker}, inp=x, out=generated)
