import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from logger import Logger, Visualizer
import numpy as np
import imageio
from sync_batchnorm import DataParallelWithCallback


def reconstruction(config, generator, kp_detector, kp_tracker, checkpoint, log_dir, dataset):
    png_gen_dir = os.path.join(log_dir, 'reconstruction/png_gen')
    png_gt_dir = os.path.join(log_dir, 'reconstruction/png_gt')
    log_dir = os.path.join(log_dir, 'reconstruction')

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector, kp_tracker=kp_tracker)
    else:
        raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_gen_dir):
        os.makedirs(png_gen_dir)

    if not os.path.exists(png_gt_dir):
        os.makedirs(png_gt_dir)

    loss_list = []
    if torch.cuda.is_available():
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)
        kp_tracker = DataParallelWithCallback(kp_tracker)

    generator.eval()
    kp_detector.eval()
    kp_tracker.eval()

    for it, x in tqdm(enumerate(dataloader)):
        if config['reconstruction_params']['num_videos'] is not None:
            if it > config['reconstruction_params']['num_videos']:
                break
        with torch.no_grad():
            predictions = []
            visualizations = []
            if torch.cuda.is_available():
                x['video'] = x['video'].cuda()
            source = x['video'][:, :, 0]
            kp_source = kp_detector(source)
            for frame_idx in range(x['video'].shape[2] // 8):
                driving = x['video'][:, :, frame_idx * 8:frame_idx * 8 + 8]
                driving_0 = driving[:, :, 0]
                kp_driving_0 = kp_detector(driving_0)
                kp_driving, _, _ = kp_tracker(driving, kp=kp_driving_0)
                out = generator(source, kp_source=kp_source, kp_driving=kp_driving)

                out['kp_source'] = kp_source
                out['kp_driving'] = kp_driving
                del out['sparse_deformed']
                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 4, 1])[0])

                visualization = Visualizer(**config['visualizer_params']).visualize_vid(source=source, driving=driving,
                                                                                        out=out, with_kp=True)
                for i in range(8):
                    visualizations.append(visualization[i])

                loss_list.append(torch.abs(out['prediction'] - driving).mean().cpu().numpy())

            predictions = np.concatenate(predictions, axis=0)
            t, h, w, c = np.shape(predictions)
            predictions = np.transpose(predictions, [1, 0, 2, 3])
            predictions = np.reshape(predictions, [h, t * w, c])
            imageio.imsave(os.path.join(png_gen_dir, x['name'][0] + '.png'), (255 * predictions).astype(np.uint8))
            
            image_name = x['name'][0] + config['reconstruction_params']['format']
            imageio.mimsave(os.path.join(log_dir, image_name), visualizations)

    print("Reconstruction loss: %s" % np.mean(loss_list))
