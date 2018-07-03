from cnn.operations import *
from cnn.utils import drop_path
import chainer
from chainer import functions as F
from chainer import links as L
from chainer import Sequential


class Cell(chainer.Chain):
    def __init__(self, genotype, prev_prev_channels, prev_channels, channels, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.pp_ch = prev_prev_channels
        self.p_ch = prev_channels
        self.ch = channels
        print(self.pp_ch, self.p_ch, self.ch)

        if reduction_prev:
            self.pre_process0 = FactorizedReduce(self.pp_ch, self.ch)
        else:
            self.pre_process0 = ReLUConvBN(self.pp_ch, self.ch, 1, 1, 0)
        self.pre_process1 = ReLUConvBN(self.p_ch, self.ch, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(self.ch, op_names, indices, concat, reduction)

    def _compile(self, ch, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = chainer.ChainList()
        for name, index in zip(op_names, indices):
            _stride = 2 if reduction and index < 2 else 1
            op = OPS[name](ch, _stride, True)
            self._ops.add_link(op)
        self._indices = indices

    def __call__(self, s0, s1, drop_prob):
        s0 = self.pre_process0(s0)
        s1 = self.pre_process1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if chainer.config.train and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return F.concat((states[i] for i in self._concat), axis=1)


class AuxiliaryHeadCIFAR(chainer.Chain):

    def __init__(self, channels, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = Sequential(
            F.relu,
            F.AveragePooling2D(ksize=5, stride=3, pad=0, cover_all=False).apply,  # image size = 2 x 2
            L.Convolution2D(channels, 128, 1, nobias=True),
            L.BatchNormalization(128),
            F.relu,
            L.Convolution2D(128, 768, 2, nobias=True),
            L.BatchNormalization(768),
            F.relu
        )
        self.classifier = L.Linear(768, num_classes)

    def __call__(self, x):
        h = self.features(x)
        h = self.classifier(F.reshape(h, (h.shape[0], -1)))
        return h


class AuxiliaryHeadImageNet(chainer.Chain):

    def __init__(self, channels, num_classes):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = Sequential(
            F.relu,
            F.AveragePooling2D(ksize=5, stride=2, pad=0, cover_all=False).apply,
            L.Convolution2D(channels, 128, 1, nobias=True),
            L.BatchNormalization(128),
            F.relu,
            L.Convolution2D(128, 768, 2, nobias=True),
            L.BatchNormalization(768),
            F.relu
        )
        self.classifier = L.Linear(768, num_classes)

    def __call__(self, x):
        h = self.features(x)
        h = self.classifier(F.reshape(h, (h.shape[0], -1)))
        return h


class NetworkCIFAR(chainer.Chain):

    def __init__(self, channels: int, num_classes: int, layers: int, auxiliary, genotype):
        super(NetworkCIFAR, self).__init__()
        self._layers: int = layers
        self._auxiliary = auxiliary

        stem_multiplier = 3
        curr_ch = stem_multiplier * channels
        self.stem = Sequential(
            L.Convolution2D(3, curr_ch, 3, pad=1, nobias=True),
            L.BatchNormalization(curr_ch)
        )

        pp_ch, p_ch, curr_ch = curr_ch, curr_ch, channels
        self.cells = chainer.ChainList()
        reduction_prev = False
        ch_to_auxiliary: int = -1
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                curr_ch *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, pp_ch, p_ch, curr_ch, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.add_link(cell)
            pp_ch, p_ch = p_ch, cell.multiplier * curr_ch
            if i == 2 * layers // 3:
                ch_to_auxiliary = p_ch
        assert ch_to_auxiliary != -1
        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(ch_to_auxiliary, num_classes)
        self.classifier = L.Linear(p_ch, num_classes)

    def __call__(self, x):
        logit_aux = None
        s0 = s1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and chainer.config.train:
                    logit_aux = self.auxiliary_head(s1)
        out = _global_average_pooling_2d(s1)
        logit = self.classifier(out)
        return logit, logit_aux


class NetworkImageNet(chainer.Chain):

    def __init__(self, channels: int, num_classes: int, layers: int, auxiliary, genotype):
        super(NetworkImageNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        self.stem0 = Sequential(
            L.Convolution2D(3, channels // 2, ksize=3, stride=2, pad=1, nobias=True),
            L.BatchNormalization(channels // 2),
            F.relu,
            L.Convolution2D(channels // 2, channels, ksize=3, stride=2, pad=1, nobias=True),
            L.BatchNormalization(channels),
        )

        self.stem1 = Sequential(
            F.relu,
            L.Convolution2D(channels, channels, 3, stride=2, pad=1, nobias=True),
            L.BatchNormalization(channels),
        )

        pp_ch, p_ch, curr_ch = channels, channels, channels

        self.cells = chainer.ChainList()
        reduction_prev = True
        ch_to_auxiliary = -1
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                curr_ch *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, pp_ch, p_ch, curr_ch, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.add_link(cell)
            pp_ch, p_ch = p_ch, cell.multiplier * curr_ch
            if i == 2 * layers // 3:
                ch_to_auxiliary = p_ch
        assert ch_to_auxiliary != -1
        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(ch_to_auxiliary, num_classes)
        self.classifier = L.Linear(p_ch, num_classes)

    def __call__(self, x):
        logit_aux = None
        s0 = self.stem0(x)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and chainer.config.train:
                    logit_aux = self.auxiliary_head(s1)
        out = _global_average_pooling_2d(s1)
        logit = self.classifier(out.view(out.size(0), -1))
        return logit, logit_aux


def _global_average_pooling_2d(x):
    n, channel, rows, cols = x.data.shape
    h = F.average_pooling_2d(x, (rows, cols), stride=1)
    h = F.reshape(h, (n, channel))
    return h
