import os
import numpy as np
import chainer
import shutil
from typing import List, Tuple
from chainercv import transforms


class AverageMeter(object):

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, top_k: Tuple[int]=(1,)):
    max_k = max(top_k)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def cutout(img: np.ndarray, length: int):
    _, h, w = img.shape
    y = np.random.randint(h)
    x = np.random.randint(w)

    y1 = np.clip(y - length // 2, 0, h)
    y2 = np.clip(y + length // 2, 0, h)
    x1 = np.clip(x - length // 2, 0, w)
    x2 = np.clip(x + length // 2, 0, w)

    img[y1: y2, x1: x2] = 0.
    return img


def padding(img: np.ndarray, pad=4):
    c, h, w = img.shape
    new_img = np.zeros((c, h + 2 * pad, w + 2 * pad))
    new_img[:, pad:h + pad, pad:w + pad] = img
    return new_img


def transform_cifar10(data: Tuple[np.ndarray, np.ndarray], mean: np.ndarray, std: np.ndarray,
                      cutout_length: int, crop_size=(32, 32), train=True):
    img, label = data
    img = img.copy()

    # Standardization
    img -= mean[:, None, None]
    img /= std[:, None, None]

    if type(crop_size) == int:
        crop_size = (crop_size, crop_size)

    if train:
        # padding
        img = padding(img, pad=4)
        # Random flip
        img = transforms.random_flip(img, x_random=True)
        # Random crop
        if crop_size != (32, 32):
            img = transforms.random_crop(img, tuple(crop_size))
        # cutout
        img = cutout(img, cutout_length)

    return img.astype(np.float32), label


def count_parameters_in_mb(model, empty_input) -> float:
    # tensor = np.empty((2, 3, 96, 96), dtype=np.float32)
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        model(empty_input)
    return sum(param.size for param in model.params())/1e6


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path: str, scripts_to_save: List[str] or None = None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
