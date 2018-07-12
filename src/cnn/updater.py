import chainer
from cnn.architect import Architect
from cnn.model_search import Network
from chainer.optimizer import Optimizer
from chainer import functions as func


class Updater(chainer.training.StandardUpdater):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._device = device
        self.model: Network = kwargs.pop('model')
        self.arch: Architect = kwargs.pop('architect')
        self.eta: float = kwargs.pop('architect')

    def _calc_loss(self, images, labels):
        optimizer_w: Optimizer = self.get_optimizer('weight')
        model = self.model
        predictions = model(images)
        loss = func.softmax_cross_entropy(predictions, labels)
        accuracy = func.accuracy(predictions, labels)
        chainer.report({'w/loss': loss}, optimizer_w)
        chainer.report({'w/accuracy': accuracy}, optimizer_w)
        return loss

    def _update_arch(self, images_val, labels_val, images_train, labels_train):
        # save_weights
        current_weights = self.model.xp.copy([p.data for p in self.model.params()])
        # reset gradients
        optimizer_a: Optimizer = self.get_optimizer('architect')
        optimizer_a.target.cleargrads()
        # save current weights
        predictions = self.model(images_train)
        loss = func.softmax_cross_entropy(predictions, labels_train)
        # calculate next gradient
        loss.backward()
        # keep architect param
        for architect_param in self.model.arch_parameters:
            architect_param.cleargrads()
        # w_k - \eta * \nabla_w L_{train}(w_k, \alpha_{k-1})
        for param in self.model.params():
            param.data -= self.eta * param.grad
        # calculate L_{val}(w_k - \eta * \nabla_w L_{train}(w_k, \alpha_{k-1}))
        optimizer_a.target.cleargrads()
        for weight_parameter in self.model.weight_parameters:
            weight_parameter.disable_update()
        for architect_param in self.model.arch_parameters:
            architect_param.enable_update()
        predictions = self.model(images_val)
        loss = func.softmax_cross_entropy(predictions, labels_val)
        loss.backward()
        accuracy = func.accuracy(predictions, labels_val)
        chainer.report({'a/loss': loss}, optimizer_a)
        chainer.report({'a/accuracy': accuracy}, optimizer_a)
        optimizer_a.update()
        # reset update mode
        for weight_parameter in self.model.weight_parameters:
            weight_parameter.enable_update()
        # reset weights
        for param, w in zip(self.model.params(), current_weights):
            param.data = w

    def update_core(self):
        # Load optimizer
        optimizer_w: Optimizer = self.get_optimizer('weight')

        # Load mini-batch
        batch_w = next(self.get_iterator('weight'))
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
        for architect_param in self.model.arch_parameters:
            architect_param.disable_update()
        optimizer_w.update(lossfun=self._calc_loss(images_w, labels_w))
        for architect_param in self.model.arch_parameters:
            architect_param.enable_update()

        # Weights update
        self._update_arch(images_a, labels_a, images_w, labels_w)
