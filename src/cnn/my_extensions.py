from __future__ import division
import numpy
from chainer.training import extension
import math


class CosineAnnealingLR(extension.Extension):
    """
    Cosine annealing shift:

    Args:
        attr (str): Name of the attribute to shift.
        t_max (int): end_epoch
        eta_min (float): Minimum learning rate. Default: 0.
        optimizer (Optimizer): Wrapped optimizer.
    """

    def __init__(self, attr: str, t_max: int, eta_min: float = 0, init=None, optimizer=None):
        self._attr = attr
        self._t_max = t_max
        self._eta_min = eta_min
        self._optimizer = optimizer
        self._last_value = None
        self._init = init

    def initialize(self, trainer):
        optimizer = self._get_optimizer(trainer)
        # ensure that _init is set
        if self._init is None:
            self._init = getattr(optimizer, self._attr)
        # resume
        if self._last_value is not None:
            value = self._last_value
        else:
            value = self._compute_next_value(trainer)
        self._update_value(optimizer, value)

    def __call__(self, trainer):
        optimizer = self._get_optimizer(trainer)
        value = self._compute_next_value(trainer)
        self._update_value(optimizer, value)

    def _compute_next_value(self, trainer):
        t_curr = trainer.updater.epoch
        return self._eta_min + (self._init - self._eta_min) * (
                    1 + math.cos(math.pi * t_curr / self._t_max)) / 2

    def _update_value(self, optimizer, value):
        setattr(optimizer, self._attr, value)
        self._last_value = value

    def serialize(self, serializer):
        self._last_value = serializer('_last_value', self._last_value)
        if isinstance(self._last_value, numpy.ndarray):
            self._last_value = numpy.asscalar(self._last_value)

    def _get_optimizer(self, trainer):
        return self._optimizer or trainer.updater.get_optimizer('main')
