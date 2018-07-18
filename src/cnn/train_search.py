import matplotlib
matplotlib.use('Agg')
import os
import time
import glob
import numpy as np
import utils
from updater import Updater
from my_extensions import CosineAnnealingLR
import logging.config
from functools import partial
import argparse
import chainer
from chainer.datasets import get_cifar10
from chainer.datasets import TransformDataset
from chainer.datasets import split_dataset_random
from chainer.iterators import MultiprocessIterator, SerialIterator
from chainer.optimizers import MomentumSGD, Adam
from chainer.optimizer import WeightDecay, GradientClipping
from chainer.training import extensions
from model_search import Network
from validate import check


def get_args():
    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--report-freq', type=float, default=50, help='report frequency')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--epoch', type=int, default=50, help='num of training epochs')
    parser.add_argument('--snapshot', type=int, default=1, help='num of snapshot epochs')
    parser.add_argument('--display', type=int, default=10, help='num of display iterations')
    parser.add_argument('--init-channels', type=int, default=16, help='num of init channels')
    parser.add_argument('--layers', type=int, default=8, help='total number of layers')
    parser.add_argument('--model-path', type=str, default='saved_models', help='path to save the model')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout-length', type=int, default=16, help='cutout length')
    parser.add_argument('--crop-size', type=int, default=32, help='cropping size')
    parser.add_argument('--drop-path-prob', type=float, default=0.3, help='drop path probability')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--out', type=str, default='results', help='path to save results')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--grad-clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--train-portion', type=float, default=0.5, help='portion of training data')
    parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
    parser.add_argument('--arch-learning-rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch-weight-decay', type=float, default=1e-3, help='weight decay for arch encoding')
    return parser.parse_args()


def get_logger(save_path):
    logging.config.fileConfig('../config/logging.conf')
    logger = logging.getLogger()
    fh = logging.FileHandler(save_path)

    # FIXME This code may be deprecated.
    # I don't know the right way to overwrite the log path while using fileConfig.
    # (Can I use a relative path in logging.conf ?)
    logger.handlers = [fh]
    return logger


def main():
    args = get_args()
    # save all python files
    args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    args.save = os.path.join(args.out, args.save)
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob(os.path.join(os.path.dirname(__file__), '*.py')))
    # save log
    logger = get_logger(save_path=os.path.join(args.save, 'train_search.log'))

    if args.gpu > 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        logger.info('gpu {} using'.format(args.gpu))
    else:
        logger.info('no gpu device available')

    np.random.seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    cifar_classes = 10
    model = Network(args.init_channels, cifar_classes, args.layers)
    if args.gpu > 0:
        model.to_gpu(args.gpu)
    empty_input = model.xp.empty(shape=(1, 3, 32, 32), dtype=model.xp.float32)
    logging.info("param size = %fMB", utils.count_parameters_in_mb(model, empty_input))

    optimizer_w = MomentumSGD(lr=args.learning_rate, momentum=args.momentum)
    optimizer_w.setup(model)
    optimizer_w.add_hook(WeightDecay(args.weight_decay))
    optimizer_a = Adam(alpha=args.arch_learning_rate, beta1=0.5, beta2=0.999)
    optimizer_a.setup(model)
    optimizer_a.add_hook(WeightDecay(args.arch_weight_decay))
    optimizer_a.add_hook(GradientClipping(10.))

    train, test = get_cifar10()
    mean = np.mean([x for x, _ in train], axis=(0, 2, 3)).astype(np.float32)
    std = np.std([x for x, _ in train], axis=(0, 2, 3)).astype(np.float32)

    train_transform = partial(utils.transform_cifar10,
                              mean=mean, std=std, cutout_length=args.cutout_length,
                              crop_size=args.crop_size, train=True)
    valid_transform = partial(utils.transform_cifar10,
                              mean=mean, std=std, cutout_length=args.cutout_length,
                              crop_size=args.crop_size, train=True)
    train_data = TransformDataset(train, train_transform)
    valid_data = TransformDataset(test, valid_transform)

    num_train = len(train_data)

    train_for_weights, train_for_arch = split_dataset_random(train_data, int(args.train_portion * num_train), seed=10)
    train_for_w_iter = MultiprocessIterator(train_for_weights, args.batch_size, repeat=True, shuffle=True, n_processes=2)
    train_for_a_iter = MultiprocessIterator(train_for_arch, args.batch_size, repeat=True, shuffle=True, n_processes=2)

    valid_iter = SerialIterator(valid_data, args.batch_size, repeat=False, shuffle=False)
    updater = Updater(
        device=args.gpu,
        model=model,
        iterator={'main': train_for_w_iter, 'architect': train_for_a_iter},
        optimizer={'main': optimizer_w, 'architect': optimizer_a},
        network_weight_decay=args.weight_decay
    )

    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(CosineAnnealingLR('lr', args.epoch, args.learning_rate_min, optimizer=optimizer_w))
    snapshot_interval = (args.snapshot, 'epoch')
    display_interval = (args.display, 'iteration')
    trainer.extend(
        extensions.observe_lr(optimizer_name='main', observation_key='lr'),
        trigger=display_interval)
    trainer.extend(
        extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}.npz'),
        trigger=snapshot_interval)
    trainer.extend(
        extensions.snapshot_object(model, 'model_epoch_{.updater.epoch}.npz'),
        trigger=snapshot_interval)
    display_items = ['epoch',
                     'iteration',
                     'lr',
                     'main/weight/loss',
                     'main/weight/accuracy',
                     'main/architect/loss',
                     'main/architect/accuracy',
                     'elapsed_time']
    trainer.extend(
        extensions.PrintReport(display_items), trigger=display_interval)
    trainer.extend(extensions.LogReport(trigger=(50, 'iteration')))
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/weight/loss', 'main/architect/loss'],
                                  file_name='loss.png', trigger=(50, 'iteration')))
        trainer.extend(
            extensions.PlotReport(['main/weight/accuracy', 'main/architect/accuracy'],
                                  file_name='accuracy.png', trigger=(50, 'iteration'))
        )
    trainer.extend(check(model, args.out), trigger=(1, 'epoch'))

    # Run the training
    print('running')
    trainer.run()


if __name__ == '__main__':
    main()
