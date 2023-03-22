#!/usr/bin/env python3

import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from cv2 import imread
import torch
import numpy as np
#from scipy.misc import imread
from scipy import linalg
from torch.autograd import Variable
from torch.nn.functional import adaptive_avg_pool2d

from gan_training.metrics.inception import InceptionV3

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'path',
    type=str,
    nargs=2,
    help=('Path to the generated images or '
          'to .npz statistic files'))
parser.add_argument(
    '--batch-size', type=int, default=64, help='Batch size to use')
parser.add_argument(
    '--dims',
    type=int,
    default=2048,
    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
    help=('Dimensionality of Inception features to use. '
          'By default, uses pool3 features'))
parser.add_argument(
    '-c',
    '--gpu',
    default='',
    type=str,
    help='GPU to use (leave blank for CPU only)')

model = None


def get_activations(images,
                    model,
                    batch_size=64,
                    dims=2048,
                    cuda=False,
                    verbose=False):

    model.eval()

    d0 = images.shape[0]
    if batch_size > d0:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = d0

    n_batches = d0 // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, dims))
    for i in range(n_batches):
        if verbose:
            print(
                '\rPropagating batch %d/%d' % (i + 1, n_batches),
                end='',
                flush=True)
        start = i * batch_size
        end = start + batch_size

        batch = torch.from_numpy(images[start:end]).type(torch.FloatTensor)
        
        with torch.no_grad():
            batch = Variable(batch)
        # batch = Variable(batch, volatile=True)  # default of GAN memory, leading to warnings
        
        if cuda:
            batch = batch.cuda()

        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)

    if verbose:
        print(' done')

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):


    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (
        diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(images,
                                    model,
                                    batch_size=64,
                                    dims=2048,
                                    cuda=False,
                                    verbose=False):

    act = get_activations(images, model, batch_size, dims, cuda, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def _compute_statistics_of_path(path, model, batch_size, dims, cuda):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        path = pathlib.Path(path)
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))

        imgs = np.array([imread(str(fn)).astype(np.float32) for fn in files])

        # Bring images to shape (B, 3, H, W)
        imgs = imgs.transpose((0, 3, 1, 2))

        # Rescale images to be between 0 and 1
        imgs /= 255

        m, s = calculate_activation_statistics(imgs, model, batch_size, dims,
                                               cuda)

    return m, s


def _compute_statistics_of_images(imgs, model, batch_size, dims, cuda):
    # values must be between 0 and 1

    # Bring images to shape (B, 3, H, W)
    if imgs.shape[1] != 3:
        imgs = imgs.transpose((0, 3, 1, 2))
    m, s = calculate_activation_statistics(imgs, model, batch_size, dims, cuda)
    return m, s


def calculate_fid_given_paths(paths, batch_size, cuda, dims):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    global model
    if model is None:
        model = _build_model(dims, cuda)

    m1, s1 = _compute_statistics_of_path(paths[0], model, batch_size, dims,
                                         cuda)
    m2, s2 = _compute_statistics_of_path(paths[1], model, batch_size, dims,
                                         cuda)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def calculate_fid_given_images(imgs0,
                               imgs1,
                               batch_size=64,
                               cuda=False,
                               dims=2048):
    global model

    if model is None:
        model = _build_model(dims, cuda)

    b0 = min(batch_size, imgs0.shape[0])
    b1 = min(batch_size, imgs1.shape[0])
    imgs0 = imgs0[:(imgs0.shape[0] // b0) * b0, ...]
    imgs1 = imgs1[:(imgs1.shape[0] // b1) * b1, ...]

    m1, s1 = _compute_statistics_of_images(imgs0, model, b0, dims, cuda)
    m2, s2 = _compute_statistics_of_images(imgs1, model, b1, dims, cuda)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def _build_model(dims, cuda):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()

    return model


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    fid_value = calculate_fid_given_paths(args.path, args.batch_size,
                                          args.gpu != '', args.dims)
    print('FID: ', fid_value)
