import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from frames_dataset import PairedDataset
from logger import Logger, Visualizer
import imageio
import numpy as np

from sync_batchnorm import DataParallelWithCallback


def animate(config, generator, kp_detector, kp_tracker, checkpoint, log_dir, dataset):
    log_dir = os.path.join(log_dir, 'animation')
    png_dir = os.path.join(log_dir, 'png')
    animate_params = config['animate_params']

    dataset = PairedDataset(initial_dataset=dataset, number_of_pairs=animate_params['num_pairs'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector, kp_tracker=kp_tracker)
    else:
        raise AttributeError("Checkpoint should be specified for mode='animate'.")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    if torch.cuda.is_available():
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)
        kp_tracker = DataParallelWithCallback(kp_tracker)

    generator.eval()
    kp_detector.eval()
    kp_tracker.eval()

    for it, x in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            predictions = []
            visualizations = []

            driving_video = x['driving_video']
            source = x['source_video'][:, :, 0]
            kp_source = kp_detector(source)

            for frame_idx in range(driving_video.shape[2] // 8):
                driving_0 = driving_video[:, :, frame_idx * 8]
                kp_driving_0 = kp_detector(driving_0)
                driving = driving_video[:, :, frame_idx * 8:frame_idx * 8 + 8]
                kp_driving, _, _ = kp_tracker(driving, kp=kp_driving_0)

                out = generator(source, kp_source=kp_source, kp_driving=kp_driving)

                out['kp_source'] = kp_source
                out['kp_driving'] = kp_driving
                del out['sparse_deformed']

                visualization = Visualizer(**config['visualizer_params']).visualize_vid(source=source,
                                                                                        driving=driving,
                                                                                        out=out, format='mp4')
                visualization = visualization
                visualizations.append(visualization)

            result_name = "-".join([x['driving_name'][0], x['source_name'][0]])
                    
            predictions = np.concatenate(predictions, axis=0)
            imageio.mimsave(os.path.join(png_dir, result_name + '.mp4'), (255 * predictions).astype(np.uint8))

            image_name = result_name + animate_params['format']
            imageio.mimsave(os.path.join(log_dir, image_name), np.concatenate(visualizations, axis=0))
