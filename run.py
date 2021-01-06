import matplotlib
matplotlib.use('Agg')

import os
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy

from frames_dataset import FramesDataset

from modules.generator import OcclusionAwareVideoGenerator
from modules.discriminator import MultiScaleImageDiscriminator, MultiScaleVideoDiscriminator
from modules.keypoint_detector import KPDetector
from modules.keypoint_tracker import KPTracker

import torch

from train import train
from reconstruction import reconstruction
from animate import animate

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "animate"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f)

    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += ' ' + strftime("%d-%m-%y %H:%M:%S", gmtime())

    generator = OcclusionAwareVideoGenerator(**config['model_params']['generator_params'],
                                             **config['model_params']['common_params'])

    if torch.cuda.is_available():
        generator.to(opt.device_ids[0])
    if opt.verbose:
        print(generator)

    image_discriminator = MultiScaleImageDiscriminator(**config['model_params']['image_discriminator_params'],
                                                       **config['model_params']['common_params'])
    if torch.cuda.is_available():
        image_discriminator.to(opt.device_ids[0])
    if opt.verbose:
        print(image_discriminator)

    video_discriminator = MultiScaleVideoDiscriminator(**config['model_params']['video_discriminator_params'],
                                                       **config['model_params']['common_params'])
    if torch.cuda.is_available():
        video_discriminator.to(opt.device_ids[0])
    if opt.verbose:
        print(video_discriminator)

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if torch.cuda.is_available():
        kp_detector.to(opt.device_ids[0])
    if opt.verbose:
        print(kp_detector)

    kp_tracker = KPTracker(**config['model_params']['kp_tracker_params'],
                           **config['model_params']['common_params'])
    if torch.cuda.is_available():
        kp_tracker.to(opt.device_ids[0])
    if opt.verbose:
        print(kp_tracker)

    dataset = FramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'])

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    if opt.mode == 'train':
        print("Training...")
        train(config, generator, image_discriminator, video_discriminator, kp_detector,
              kp_tracker, opt.checkpoint, log_dir, dataset, opt.device_ids)
    elif opt.mode == 'reconstruction':
        print("Reconstruction...")
        reconstruction(config, generator, kp_detector, kp_tracker, opt.checkpoint, log_dir, dataset)
    elif opt.mode == 'animate':
        print("Animate...")
        animate(config, generator, kp_detector, kp_tracker, opt.checkpoint, log_dir, dataset)
