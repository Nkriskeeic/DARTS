import chainer
from chainer import links as L
from chainer import functions as F
from chainer import Sequential

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: F.AveragePooling2D(ksize=3, stride=stride, pad=1, cover_all=False).apply,
    'max_pool_3x3': lambda C, stride, affine: F.MaxPooling2D(ksize=3, stride=stride, pad=1, cover_all=False).apply,
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine: Sequential(
        F.relu,
        L.Convolution2D(C, C, (1, 7), stride=(1, stride), padding=(0, 3), nobias=True),
        L.Convolution2D(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), nobias=True),
        L.BatchNormalization(C, use_gamma=affine, use_beta=affine)
    ),
}


class ReLUConvBN(chainer.Chain):
    """
    活性化(ReLU)→conv→BN
    元実装: https://github.com/quark0/darts/blob/master/cnn/operations.py ReLUConvBNより
    """
    def __init__(self, in_channels, out_channels, k_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = Sequential(
            F.relu,
            L.Convolution2D(in_channels, out_channels, k_size, stride, padding, nobias=True),
            # affineがFalseだとgammaとbetaはそれぞれ1と0になる
            L.BatchNormalization(out_channels, use_gamma=affine, use_beta=affine),
        )

    def __call__(self, x):
        return self.op(x)


class DilConv(chainer.Chain):
    """
    活性化(ReLU)→depth-wise dilated conv→point-wise convolution→BN
    元実装: https://github.com/quark0/darts/blob/master/cnn/operations.py DilConvより
    TODO: これってSeparable ConvolutionのDilated版だよね?
    """
    def __init__(self, in_channels, out_channels, k_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = Sequential(
            F.relu,
            GroupedDilConv(in_channels, out_channels, k_size, stride, padding, dilation,
                           groups=in_channels),
            L.Convolution2D(in_channels, out_channels, ksize=1, padding=0, nobias=True),
            L.BatchNormalization(out_channels, use_gamma=affine, use_beta=affine),
        )

    def __call__(self, x):
        return self.op(x)


class GroupedDilConv(chainer.Chain):
    """
    Grouped Dilated Convolution実装．chainerを軽く探したけど見つけられなかったので取り敢えず．native実装があれば変更．
    TODO: native実装があるかを調べる
    """
    def __init__(self, in_channels, out_channels, k_size, stride, padding, dilation, groups):
        super(GroupedDilConv, self).__init__()
        assert in_channels % groups == 0 and out_channels % groups == 0
        grouped_in_channels = in_channels // groups
        grouped_out_channels = out_channels // groups
        self.groups = groups
        with self.init_scope():
            self.c = L.DilatedConvolution2D(grouped_in_channels, grouped_out_channels,
                                            k_size, stride, padding, dilation, nobias=True)

    def __call__(self, x):
        _n, _c, _h, _w = x.shape
        assert _c % self.groups
        h = F.reshape(x, (_n * self.groups, _c // self.groups, _h, _w))
        h = self.c(h)
        return F.reshape(h, (_n, _c, h, _w))


class SepConv(chainer.Chain):
    """
    Separable Convolution
    ReLU→depth wise→point wise→BN→ReLU→depth wise→point wise→BN
    元実装: https://github.com/quark0/darts/blob/master/cnn/operations.py SepConvより
    """
    def __init__(self, in_channels, out_channels, k_size, stride, padding, affine=True):
        assert out_channels % in_channels == 0
        super(SepConv, self).__init__()
        self.op = Sequential(
            F.relu,
            L.DepthwiseConvolution2D(in_channels, 1, k_size, stride, padding, nobias=True),
            L.Convolution2D(in_channels, in_channels, ksize=1, pad=0, nobias=True),
            L.BatchNormalization(in_channels, use_gamma=affine, use_beta=affine),
            F.relu,
            L.DepthwiseConvolution2D(in_channels, 1, k_size, stride=1, pad=padding, nobias=True),
            L.Convolution2D(in_channels, out_channels, ksize=1, pad=0, nobias=True),
            L.BatchNormalization(out_channels, use_gamma=affine, use_beta=affine),
        )

    def __call__(self, x):
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

    def __call__(self, x):
        if self.stride == 1:
            return x * 0
        return x[:, :, ::self.stride, ::self.stride] * 0


class FactorizedReduce(chainer.Chain):
    """
    元実装: https://github.com/quark0/darts/blob/master/cnn/operations.py FactorizedReduceより
    """
    def __init__(self, in_channels, out_channels, affine=True):
        super(FactorizedReduce, self).__init__()
        assert out_channels % 2 == 0
        with self.init_scope():
            self.conv_1 = L.Convolution2D(in_channels, out_channels // 2, ksize=1, stride=2, pad=0, nobias=True)
            self.conv_2 = L.Convolution2D(in_channels, out_channels // 2, ksize=1, stride=2, pad=0, nobias=True)
            self.bn = L.BatchNormalization(out_channels, use_gamma=affine, use_beta=affine)

    def __call__(self, x):
        h = F.relu(x)
        h1 = self.conv_1(h)
        h2 = self.conv_2(h[:, :, 1:, 1:])
        out = F.concat((h1, h2), axis=1)
        return self.bn(out)
