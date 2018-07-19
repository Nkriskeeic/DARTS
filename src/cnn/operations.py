import chainer
from chainer import links
from chainer import functions as func
from chainer import Sequential
from functools import partial

OPS = {
    'none': lambda _ch, _stride, _affine: Zero(_stride),
    'avg_pool_3x3': lambda _ch, _stride, _affine: partial(func.average_pooling_2d, ksize=3, stride=_stride, pad=1),
    'max_pool_3x3': lambda _ch, _stride, _affine: partial(func.max_pooling_2d, ksize=3, stride=_stride, pad=1, cover_all=False),
    'skip_connect': lambda _ch, _stride, _affine: Identity() if _stride == 1 else FactorizedReduce(_ch, _ch, affine=_affine),
    'sep_conv_3x3': lambda _ch, _stride, _affine: SepConv(_ch, _ch, 3, _stride, 1, affine=_affine),
    'sep_conv_5x5': lambda _ch, _stride, _affine: SepConv(_ch, _ch, 5, _stride, 2, affine=_affine),
    'sep_conv_7x7': lambda _ch, _stride, _affine: SepConv(_ch, _ch, 7, _stride, 3, affine=_affine),
    'dil_conv_3x3': lambda _ch, _stride, _affine: DilConv(_ch, _ch, 3, _stride, 2, 2, affine=_affine),
    'dil_conv_5x5': lambda _ch, _stride, _affine: DilConv(_ch, _ch, 5, _stride, 4, 2, affine=_affine),
    'conv_7x1_1x7': lambda _ch, _stride, _affine: Sequential(
        func.relu,
        links.Convolution2D(_ch, _ch, (1, 7), stride=(1, _stride), padding=(0, 3), nobias=True),
        links.Convolution2D(_ch, _ch, (7, 1), stride=(_stride, 1), padding=(3, 0), nobias=True),
        links.BatchNormalization(_ch, use_gamma=_affine, use_beta=_affine)
    ),
}


class ReLUConvBN(chainer.Chain):
    """
    活性化(ReLU)→conv→BN
    元実装: https://github.com/quark0/darts/blob/master/cnn/operations.py ReLUConvBNより
    """
    def __init__(self, in_channels, out_channels, k_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        with self.init_scope():
            self.op = Sequential(
                func.relu,
                links.Convolution2D(in_channels, out_channels, k_size, stride, padding, nobias=True),
                # affineがFalseだとgammaとbetaはそれぞれ1と0になる
                links.BatchNormalization(out_channels, use_gamma=affine, use_beta=affine),
            )

    def __call__(self, x) -> chainer.Variable:
        return self.op(x)


class DilConv(chainer.Chain):
    """
    活性化(ReLU)→depth-wise dilated conv→point-wise convolution→BN
    元実装: https://github.com/quark0/darts/blob/master/cnn/operations.py DilConvより
    """
    def __init__(self, in_channels, out_channels, k_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        with self.init_scope():
            self.op = Sequential(
                func.relu,
                links.Convolution2D(in_channels, out_channels, ksize=k_size, stride=stride,
                                    pad=padding, dilate=dilation, groups=in_channels, nobias=True),
                links.Convolution2D(in_channels, out_channels, ksize=1, pad=0, nobias=True),
                links.BatchNormalization(out_channels, use_gamma=affine, use_beta=affine),
            )

    def __call__(self, x) -> chainer.Variable:
        return self.op(x)


class SepConv(chainer.Chain):
    """
    Separable Convolution
    ReLU→depth wise→point wise→BN→ReLU→depth wise→point wise→BN
    元実装: https://github.com/quark0/darts/blob/master/cnn/operations.py SepConvより
    """
    def __init__(self, in_channels, out_channels, k_size, stride, padding, affine=True):
        assert out_channels % in_channels == 0
        super(SepConv, self).__init__()
        with self.init_scope():
            self.op = Sequential(
                func.relu,
                links.Convolution2D(in_channels, in_channels, k_size, stride, padding, groups=in_channels, nobias=True),
                links.Convolution2D(in_channels, in_channels, ksize=1, pad=0, nobias=True),
                links.BatchNormalization(in_channels, use_gamma=affine, use_beta=affine),
                func.relu,
                links.Convolution2D(in_channels, in_channels, k_size, stride=1, pad=padding, groups=in_channels, nobias=True),
                links.Convolution2D(in_channels, out_channels, ksize=1, pad=0, nobias=True),
                links.BatchNormalization(out_channels, use_gamma=affine, use_beta=affine),
            )

    def __call__(self, x) -> chainer.Variable:
        return self.op(x)


class Identity(chainer.Chain):
    """
    恒等写像
    元実装: https://github.com/quark0/darts/blob/master/cnn/operations.py Identityより
    """
    def __init__(self):
        super(Identity, self).__init__()

    def __call__(self, x):
        return x


class Zero(chainer.Chain):
    """
    (n, c, h/stride, w/stride) の大きさの0を返す
    元実装: https://github.com/quark0/darts/blob/master/cnn/operations.py Zeroより
    """
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def __call__(self, x) -> chainer.Variable:
        if self.stride == 1:
            return x * 0
        return x[:, :, ::self.stride, ::self.stride] * 0


class FactorizedReduce(chainer.Chain):
    """
    元実装: https://github.com/quark0/darts/blob/master/cnn/operations.py FactorizedReduceより
    strideが2なので情報損失を防ぐために1, 1ずらしたものも混ぜる
    """
    def __init__(self, in_channels, out_channels, affine=True):
        super(FactorizedReduce, self).__init__()
        assert out_channels % 2 == 0
        with self.init_scope():
            self.conv_1 = links.Convolution2D(in_channels, out_channels // 2, ksize=1, stride=2, pad=0, nobias=True)
            self.conv_2 = links.Convolution2D(in_channels, out_channels // 2, ksize=1, stride=2, pad=0, nobias=True)
            self.bn = links.BatchNormalization(out_channels, use_gamma=affine, use_beta=affine)

    def __call__(self, x) -> chainer.Variable:
        h = func.relu(x)
        h1 = self.conv_1(h)
        h2 = self.conv_2(h[:, :, 1:, 1:])
        out = func.concat((h1, h2), axis=1)
        return self.bn(out)
