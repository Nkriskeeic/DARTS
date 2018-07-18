from operations import *
from genotypes import PRIMITIVES
from genotypes import Genotype
import chainer
from chainer import Sequential
from chainer import links
from chainer import functions as func
from typing import List, Tuple


class MixedOp(chainer.Chain):
    def __init__(self, channels: int, stride):
        super(MixedOp, self).__init__()
        with self.init_scope():
            self._ops = chainer.ChainList()
            for primitive in PRIMITIVES:
                op = OPS[primitive](channels, stride, False)
                if 'pool' in primitive:
                    op = Sequential(op, links.BatchNormalization(channels, use_gamma=False, use_beta=False))
                self._ops.add_link(op)

    def __call__(self, x, weights):
        h = 0
        for w, op in zip(weights, self._ops):
            h += w * op(x)
        return h


class Cell(chainer.Chain):
    def __init__(self, steps: int, multiplier: int, prev_prev_channels: int, prev_channels: int, channels: int,
                 reduction: bool, reduction_prev: bool):
        super(Cell, self).__init__()
        with self.init_scope():
            self.reduction = reduction
            self.pp_ch = prev_prev_channels
            self.p_ch = prev_channels
            self.ch = channels

            if reduction_prev:
                self.pre_process0 = FactorizedReduce(self.pp_ch, self.ch, affine=False)
            else:
                self.pre_process0 = ReLUConvBN(self.pp_ch, self.ch, 1, 1, 0, affine=False)
            self.pre_process1 = ReLUConvBN(self.p_ch, self.ch, 1, 1, 0, affine=False)
            self._steps = steps
            self._multiplier = multiplier

            self._ops = chainer.ChainList()
            self._bns = chainer.ChainList()
            for i in range(self._steps):
                for j in range(2 + i):
                    stride = 2 if reduction and j < 2 else 1
                    op = MixedOp(channels, stride)
                    self._ops.add_link(op)

    def __call__(self, s0, s1, weights):
        s0 = self.pre_process0(s0)
        s1 = self.pre_process1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = 0
            for j, h in enumerate(states):
                s += self._ops[offset + j](h, weights[offset + j])
            offset += len(states)
            states.append(s)
        return func.concat(states[-self._multiplier:], axis=1)


class Attention(chainer.Chain):
    def __init__(self, num_combinations: int, num_operations: int):
        super(Attention, self).__init__()
        with self.init_scope():
            self.attention = chainer.Parameter(
                initializer=chainer.initializers.Normal(scale=1e-3),
                shape=(num_combinations, num_operations)
            )


class Network(chainer.Chain):

    def __init__(self, channels: int, num_classes: int, layers: int,
                 steps: int=4, multiplier: int=4, stem_multiplier: int=3,
                 criterion=func.softmax_cross_entropy):
        super(Network, self).__init__()

        # 初項2 (入力数), 項数steps, 公差1
        k = int((steps + 1) * (2 + steps / 2))
        num_ops = len(PRIMITIVES)

        self._steps = steps
        self._multiplier = multiplier

        curr_ch = stem_multiplier * channels

        with self.init_scope():
            self.stem = Sequential(
                links.Convolution2D(in_channels=3, out_channels=curr_ch, ksize=3, pad=1, nobias=True),
                links.BatchNormalization(curr_ch)
            )

            pp_ch, p_ch, curr_ch = curr_ch, curr_ch, channels
            self.cells = chainer.ChainList()
            reduction_prev = False

            for i in range(layers):
                if i in [layers // 3, 2 * layers // 3]:
                    curr_ch *= 2
                    reduction = True
                else:
                    reduction = False
                cell = Cell(steps, multiplier, pp_ch, p_ch, curr_ch, reduction, reduction_prev)
                reduction_prev = reduction
                self.cells.add_link(cell)
                pp_ch, p_ch = p_ch, multiplier * curr_ch

            self.classifier = links.Linear(p_ch, num_classes)
        # params()ではalphaを取得したくないのでinit_scopeの外に出す
        # 元実装に従うけどこれはresumeしない前提なのかな
        self.alphas_normal = Attention(k, num_ops)
        self.alphas_reduce = Attention(k, num_ops)
        self._arch_parameters = [self.alphas_normal, self.alphas_reduce]

        self._criterion = criterion

    def __call__(self, x):
        s0 = s1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                attention = self.alphas_reduce.attention
            else:
                attention = self.alphas_normal.attention
            shape = attention.shape
            weights = func.softmax(attention.reshape((1, -1)), axis=1).reshape(shape)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self._global_average_pooling(s1)
        logit = self.classifier(out)
        return logit

    def _loss(self, x, target):
        logit = self(x)
        return self._criterion(logit, target)

    @property
    def arch_parameters(self) -> List[chainer.Chain]:
        return self._arch_parameters

    def genotype(self) -> Genotype:
        # Cellの結合方法を定義したGenotypeを返す
        def _parse(weights: chainer.Parameter) -> List[Tuple[str, int]]:
            gene = []
            n = 2
            start = 0
            none_index = PRIMITIVES.index('none')
            for i in range(self._steps):
                end = start + n
                w = weights[start:end]
                # 最も確率の高いedgeへのリンクを2つ取得する (何もしない操作の確率は無視)
                edges = sorted(range(i + 2),
                               key=lambda x: max(w[x][op] for op in range(len(w[x])) if op != none_index),
                               reverse=True)[:2]
                # 2つのリンクの中で最も確率の高いoperationを取得する (何もしない操作の確率は無視)
                for j in edges:
                    k_best = None
                    for k in range(len(w[j])):
                        if k != none_index and (k_best is None or w[j][k] > w[j][k_best]):
                            k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(func.softmax(self.alphas_normal.attention, axis=-1).data)
        gene_reduce = _parse(func.softmax(self.alphas_reduce.attention, axis=-1).data)

        concat = list(range(2 + self._steps - self._multiplier, self._steps + 2))
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype

    @staticmethod
    def _global_average_pooling(x: chainer.Variable):
        return func.mean(x, axis=(2, 3))

    def to_gpu(self, device=None):
        super().to_gpu(device=device)
        self.alphas_normal.to_gpu(device=device)
        self.alphas_reduce.to_gpu(device=device)
