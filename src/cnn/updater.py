import chainer
from model_search import Network
from chainer.optimizer import Optimizer
from chainer.optimizers import MomentumSGD
from chainer import functions as func
from typing import List, Union
import numpy as np
import cupy as cp

NDARRAY = Union[np.ndarray, cp.ndarray]
NDARRAY_LIST = List[NDARRAY]


class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.model: Network = kwargs.pop('model')
        self.network_weight_decay: float = kwargs.pop('network_weight_decay')
        super().__init__(*args, **kwargs)
        optimizer: MomentumSGD = self.get_optimizer('main')
        self.eta: float = optimizer.__getattribute__('lr')

    def _calc_loss(self, images, labels):
        model = self.model
        predictions = model(images)
        loss = func.softmax_cross_entropy(predictions, labels)
        accuracy = func.accuracy(predictions, labels)
        chainer.report({'weight/loss': loss, 'weight/accuracy': accuracy}, self.model)
        return loss

    def _update_arch(self, images_val, labels_val, images_train, labels_train):
        model = self.model
        # save_weights
        current_weights = [p.data for p in model.params()]
        # reset gradients
        optimizer_a: Optimizer = self.get_optimizer('architect')
        optimizer_a.target.cleargrads()
        # save current weights
        predictions = model(images_train)
        loss = func.softmax_cross_entropy(predictions, labels_train)
        del predictions
        # calculate next gradient
        loss.backward()
        del loss
        # w_k - \eta * \nabla_w L_{train}(w_k, \alpha_{k-1})
        # model.params() returns weights (not alphas)
        for param in model.params():
            param.data -= self.eta * param.grad
        # calculate L_{val}(w_k - \eta * \nabla_w L_{train}(w_k, \alpha_{k-1}))
        optimizer_a.target.cleargrads()
        predictions = model(images_val)
        loss = func.softmax_cross_entropy(predictions, labels_val)
        accuracy = func.accuracy(predictions, labels_val)
        del predictions
        chainer.report({'architect/loss': loss, 'architect/accuracy': accuracy}, model)

        arch_grads: List[chainer.Variable] = chainer.grad([loss], [p for ap in model.arch_parameters
                                                                   for p in ap.params()], set_grad=True)
        model_grads: List[chainer.Variable] = chainer.grad([loss], [p for p in model.params()])
        vector: NDARRAY = [dt.data + self.network_weight_decay * t.data for dt, t in zip(model_grads, model.params())]
        del model_grads
        self.model.cleargrads()
        implicit_grads: NDARRAY_LIST = self._hessian_vector_product(vector, images_train, labels_train)
        del vector
        for g, ig in zip(arch_grads, implicit_grads):
            g.data -= self.eta * ig
        optimizer_a.update()
        del implicit_grads
        del arch_grads
        # reset weights
        for param, w in zip(model.params(), current_weights):
            param.data = w
        del current_weights

    def _hessian_vector_product(self, vector: NDARRAY_LIST, images: NDARRAY, labels: NDARRAY, r=1e-2) -> NDARRAY_LIST:
        norm = 0.
        model = self.model
        for v in vector:
            norm += model.xp.sum(v ** 2)
        norm **= 0.5
        epsilon = r / norm
        # calculate \nabla_\alpha L_{train}(w^+, \alpha)
        for p, v in zip(model.params(), vector):
            p.data += epsilon * v
        loss = func.softmax_cross_entropy(model(images), labels)
        grads_p = chainer.grad([loss], [p for ap in self.model.arch_parameters for p in ap.params()])

        # calculate \nabla_\alpha L_{train}(w^-, \alpha)
        for p, v in zip(model.params(), vector):
            p.data -= 2 * epsilon * v
        loss = func.softmax_cross_entropy(model(images), labels)
        grads_n = chainer.grad([loss], [p for ap in self.model.arch_parameters for p in ap.params()])

        # Reset model weight
        for p, v in zip(model.params(), vector):
            p += epsilon * v
        return [(x.data - y.data) / (2 * epsilon) for x, y in zip(grads_p, grads_n)]

    def update_core(self):
        # Load optimizer
        optimizer_w: Optimizer = self.get_optimizer('main')

        # Load mini-batch
        batch_w = next(self.get_iterator('main'))
        batch_a = next(self.get_iterator('architect'))
        images_w = [h for h, _ in batch_w]
        labels_w = [l for _, l in batch_w]
        images_a = [h for h, _ in batch_a]
        labels_a = [l for _, l in batch_a]
        images_w = self.converter(images_w, self.device)
        labels_w = self.converter(labels_w, self.device)
        images_a = self.converter(images_a, self.device)
        labels_a = self.converter(labels_a, self.device)

        # Weight update (not update architect parameters)
        print('update weight')
        for architect_param in self.model.arch_parameters:
            architect_param.disable_update()
        optimizer_w.update(self._calc_loss, images_w, labels_w)
        for architect_param in self.model.arch_parameters:
            architect_param.enable_update()

        # Architecture update
        print('update architect')
        self._update_arch(images_a, labels_a, images_w, labels_w)
