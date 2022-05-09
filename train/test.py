import os
import pickle
import time
from os.path import join, dirname

import cv2
import lmdb
import numpy as np
import torch
import torch.nn as nn

from data.utils import normalize, normalize_reverse
from model import Model
from .metrics import psnr_calculate, ssim_calculate
from .utils import AverageMeter, img2video


def test(config, logger):
    # load model with checkpoint
    if not config.test_only:
        config.test_checkpoint = join(logger.save_dir, 'model_best.pth.tar')
    if config.test_save_dir is None:
        config.test_save_dir = logger.save_dir
    model = Model(config).cuda()

    checkpoint_path = config.test_checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda())

    logger('Load %s as the checkpoint'%(checkpoint_path))

    model.load_state_dict(checkpoint['state_dict'])

    ds_name = config.dataset
    logger('{} results generating ...'.format(ds_name), prefix='\n')
    test_lmdb(config, logger, model)



def test_lmdb(config, logger, model):
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    timer = AverageMeter()
    results_register = set()
    if config.dataset == 'gopro_ds':
        H, W, C = 540, 960, 3
    elif config.dataset == 'gopro_ori':
        H, W, C = 720, 1280, 3
    else:
        raise ValueError

    data_test_path = config.test_input_path
    data_test_gt_path = config.test_gt_path
    data_test_info_path = config.test_info_path

    if not hasattr(config, 'past_frames'):
        config.past_frames = 0
    if not hasattr(config, 'future_frames'):
        config.future_frames = 0

    env_blur = lmdb.open(data_test_path, map_size=1099511627776)
    env_gt = lmdb.open(data_test_gt_path, map_size=1099511627776)
    txn_blur = env_blur.begin()
    txn_gt = env_gt.begin()

    with open(data_test_info_path, 'rb') as f:
        seqs_info = pickle.load(f)
    for seq_idx in range(seqs_info['num']):
        seq_length = seqs_info[seq_idx]['length']
        seq = '{:03d}'.format(seq_idx)
        logger('Start generating {}-th sequence results...'.format(seq))
        save_dir = join(config.test_save_dir, seq)
        os.makedirs(save_dir, exist_ok=True)
        start = 0
        end = seq_length if config.test_frames == -1 else config.test_frames
        my_psnr_list = []
        my_ssim_list = []

        while (True):
            input_seq = []
            label_seq = []
            for frame_idx in range(start, end):
                code = '%03d_%08d' % (seq_idx, frame_idx)
                code = code.encode()
                blur_img = txn_blur.get(code)
                blur_img = np.frombuffer(blur_img, dtype='uint8')
                blur_img = blur_img.reshape(H, W, C).transpose((2, 0, 1))[np.newaxis, :]
                gt_img = txn_gt.get(code)
                gt_img = np.frombuffer(gt_img, dtype='uint8')
                gt_img = gt_img.reshape(H, W, C)
                input_seq.append(blur_img)
                label_seq.append(gt_img)

            input_seq = np.concatenate(input_seq)[np.newaxis, :]
            model.eval()
            logger('Shape of the current input sequence: %s'%(str(input_seq.shape)))
            with torch.no_grad():
                input_seq = normalize(torch.from_numpy(input_seq).float().cuda(), centralize=config.centralize,
                                      normalize=config.normalize)
                time_start = time.time()
                output_seq = model([input_seq, ])
                output_seq = output_seq[-1]
                if isinstance(output_seq, (list, tuple)):
                    output_seq = output_seq[0]

                output_seq = output_seq.squeeze(dim=0)
                timer.update((time.time() - time_start) / len(output_seq), n=len(output_seq))


            logger('Evaluating PSNR and SSIM...')
            for frame_idx in range(config.past_frames, end - start - config.future_frames):
                blur_img = input_seq.squeeze()[frame_idx]
                blur_img = normalize_reverse(blur_img, centralize=config.centralize, normalize=config.normalize)
                blur_img = blur_img.detach().cpu().numpy().transpose((1, 2, 0)).astype(np.uint8)
                blur_img_path = join(save_dir, '{:08d}_input.png'.format(frame_idx + start))
                gt_img = label_seq[frame_idx]
                gt_img_path = join(save_dir, '{:08d}_gt.png'.format(frame_idx + start))
                deblur_img = output_seq[frame_idx - config.past_frames]
                deblur_img = normalize_reverse(deblur_img, centralize=config.centralize, normalize=config.normalize)
                deblur_img = deblur_img.detach().cpu().numpy().transpose((1, 2, 0))
                deblur_img = np.clip(deblur_img, 0, 255).astype(np.uint8)
                deblur_img_path = join(save_dir, '{:08d}_{}.png'.format(frame_idx + start, config.model.lower()))
                if config.test_save_img is True:
                    cv2.imwrite(blur_img_path, blur_img)
                    cv2.imwrite(gt_img_path, gt_img)
                    cv2.imwrite(deblur_img_path, deblur_img)
                if deblur_img_path not in results_register:
                    results_register.add(deblur_img_path)
                    psnr = psnr_calculate(deblur_img, gt_img)
                    ssim = ssim_calculate(deblur_img, gt_img)
                    PSNR.update(psnr)
                    SSIM.update(ssim)
                    my_psnr_list.append(psnr)
                    my_ssim_list.append(ssim)
            
            if end == seq_length:
                break
            else:
                start = end - config.future_frames - config.past_frames
                end = start + config.test_frames
                if end > seq_length:
                    end = seq_length
                    start = end - config.test_frames
            logger('Finish writing the image to the file...')

        if config.video:
            logger('Generate seq {} video result...'.format(seq))
            marks = ['Input', config.model, 'GT']
            path = dirname(save_dir)
            frame_start = config.past_frames
            frame_end = seq_length - config.future_frames
            img2video(path=path, size=(3 * W, 1 * H), seq=seq, frame_start=frame_start, frame_end=frame_end,
                      marks=marks, fps=10)

    logger('Test images : {}'.format(PSNR.count), prefix='\n')
    logger('Test PSNR : {}'.format(PSNR.avg))
    logger('Test SSIM : {}'.format(SSIM.avg))
    logger('Average time per image: {}'.format(timer.avg))
