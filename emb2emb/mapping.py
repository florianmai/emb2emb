import torch
from torch import nn
from random import shuffle
from abc import ABC, abstractmethod


class Mapping(nn.Module, ABC):
    """Instances of this class map from an input embedding to an 
    output embedding of the same dimensionality.

    :param embedding_size: The dimensionality of the input (and output) embedding.
    :type embedding_size: int
    """

    def __init__(self, embedding_size):
        super(Mapping, self).__init__()
        self.embedding_size = embedding_size

    @abstractmethod
    def forward(self, embedding):
        """Applies the mapping to a batch of input embeddings, and returns the output
        as a result.

        :param embeddings: A batch of input embeddings. A tensor of size 
            [batch_size x embedding_size].
        :type embeddings: torch.tensor
        :return: A batch of output embeddings denoting the result of the mapping.
            A tensor of size [batch_size x embedding_size].
        :rtype: torch.tensor
        """


class Id(Mapping):
    """
    A mapping that forwards the input to the output, i.e., applies the identity function.

    :param embedding_size: The dimensionality of the input (and output) embedding.
    """

    def __init__(self, embedding_size):
        super(Id, self).__init__(embedding_size)

    def forward(self, embeddings):
        """Returns the input as is..

        :param embeddings: A batch of input embeddings. A tensor of size 
            [batch_size x embedding_size].
        :type embeddings: torch.tensor
        :return: The same batch of embeddings that was given as input.
            A tensor of size [batch_size x embedding_size].
        :rtype: torch.tensor
        """
        return embeddings


class OffsetNet(Mapping):
    """
    The OffsetNet mapping iteratively computes an offset vectors to add to the input embedding:

    .. math::
        x_i = x_{i - 1} + MLP_i(x_{i - 1})

    where :math:`F(x)` is an MLP with an activation function (`SELU` by default), :math:`x_0 = x`
    and :math:`0 \leq i \leq n_layers`.

    :param embedding_size: The dimensionality of the input (and output) embedding
    :type embedding_size: int
    :param n_layers: Number of offset vectors to compute and add to the input
    :type n_layers: int
    :param nonlinearity: A PyTorch activation function such as ReLU or SELU
    :type nonlinearity: torch.nn.Module
    :param dropout: PyTorch module for performing dropout (e.g. `Dropout` or `AlphaDropout`)
    :type dropout: torch.nn.Module
    :param dropout_p: Dropout rate to apply to input at each layer. (0, 1)
    :type dropout_p: float
    :param offset_dropout_p: Dropout rate to apply to offset vectors. (0, 1)
    :type offset_dropout_p: float
    """

    def __init__(self, embedding_size, n_layers, nonlinearity=nn.SELU, dropout=nn.AlphaDropout,
                 dropout_p=0, offset_dropout_p=0.):
        super(OffsetNet, self).__init__(embedding_size)

        self.dropout_p = dropout_p
        self.offset_dropout_p = offset_dropout_p
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.hidden_size = embedding_size
        activation = nonlinearity

        inlayers = []
        outlayers = []
        activations = []
        dropouts = []
        offset_dropouts = []
        for _ in range(n_layers):
            inlayers.append(nn.Linear(self.hidden_size, self.hidden_size))
            activations.append(activation())
            outlayers.append(nn.Linear(self.hidden_size, self.hidden_size))

            if dropout_p != 0:
                dropouts.append(dropout(p=dropout_p))

            if offset_dropout_p > 0.:
                offset_dropouts.append(nn.Dropout(p=offset_dropout_p))

        self.mapping_in = nn.ModuleList(inlayers)
        self.mapping_out = nn.ModuleList(outlayers)
        self.activations = nn.ModuleList(activations)
        self.dropouts = nn.ModuleList(dropouts)
        self.offset_dropouts = nn.ModuleList(offset_dropouts)

    def forward(self, embeddings):

        x = embeddings
        for i in range(self.n_layers):
            if self.dropout_p > 0:
                x = self.dropouts[i](x)

            # compute offset vector
            offset = self.mapping_in[i](x)
            offset = self.activations[i](offset)
            offset = self.mapping_out[i](offset)

            if self.offset_dropout_p > 0.:
                offset = self.offset_dropouts[i](offset)

            # and add it
            x = x + offset

        return x


class ResNet(Mapping):
    """
    The (1-layer) ResNet architecture can be characterized as plugging another 
    non-linearity on top of the output of `OffsetNet`. Because the output of the mapping
    requires infinite range, this necessitates another linear transformation after that.
    Hence the output :math:`y` is computed as:

    .. math::
        y = \boldsymbol{W}\sigma(\operatorname{OffsetNet}(x))

    :param embedding_size: The dimensionality of the input (and output) embedding
    :type embedding_size: int
    :param n_layers: Number of offset vectors to compute in the `OffsetNet` component
    :type n_layers: int
    :param nonlinearity: A PyTorch activation function such as ReLU or SELU
    :type nonlinearity: torch.nn.Module
    :param dropout: PyTorch module for performing dropout (e.g. `Dropout` or `AlphaDropout`)
    :type dropout: torch.nn.Module
    :param dropout_p: Dropout rate to apply to input at each layer. :math:`(0, 1)`
    :type dropout_p: float
    :param offset_dropout_p: Dropout rate to apply to offset vectors. :math:`(0, 1)`
    :type offset_dropout_p: float
    """

    def __init__(self, embedding_size, n_layers, nonlinearity=nn.SELU, dropout=nn.AlphaDropout,
                 dropout_p=0, offset_dropout_p=0.):
        super(ResNet, self).__init__(embedding_size)

        self.offsetnet = OffsetNet(embedding_size, n_layers, nonlinearity=nonlinearity, dropout=dropout,
                                   dropout_p=dropout_p, offset_dropout_p=offset_dropout_p)
        self.final_activation = nonlinearity()
        self.final_layer = nn.Linear(embedding_size, embedding_size)

    def forward(self, embeddings):
        x = self.offsetnet(embeddings)
        x = self.final_activation(x)
        x = self.final_layer(x)
        return x


class MeanOffsetVectorMLP(Mapping):
    """
    This mapping applies two fixed offset vectors, scaled by a fixed factor. 
    The vectors are computed from two sets :math:`X,Y` of texts at initialization time.
    A random subsample of the two sets are encoded into fixed-sized embeddings and the
    averages over vectors from each set is computed, :math:`x_{avg}, y_{avg}`.

    At inference time, the output vector :math:`y` is computed from the input vector :math:`x` as:

    .. math::
        y = x + factor * (- x_{avg} + y_{avg})

    Intuitively, this removes the common characteristics among samples in set :math:`X`,
    and adds those of samples in set :math:`Y` instead.

    :param embedding_size: The dimensionality of the input (and output) embedding
    :type embedding_size: int
    :param factor: Used to scale the computed set-averages.
    :type factor: float
    :param encoder: Used for obtaining a fixed-size embedding from textual input.
    :type encoder: emb2emb.encoding.Encoder
    :param X: Set :math:`X`
    :type X: list(str)
    :param Y: Set :math:`Y`
    :type Y: list(str)
    :param num_samples: Size of the subsample of sets :math:`X,Y`
    :type num_samples: int
    :param bsize: Batch size for encoding samples with the encoder
    :type bsize: int
    """

    def __init__(self, embedding_size, factor, encoder, X, Y, num_samples=1000, bsize=16):
        super(MeanOffsetVectorMLP, self).__init__(embedding_size)
        self.factor = torch.tensor(factor)

        # shuffle to take a random sample of size 'num_samples'
        shuffle(X)
        shuffle(Y)

        x_mean = None
        y_mean = None
        with torch.no_grad():
            for idx in range(0, num_samples, bsize):

                x_emb = encoder(X[idx: idx + bsize])
                if x_mean is None:
                    x_mean = x_emb.sum(dim=0)
                else:
                    x_mean = x_mean + x_emb.sum(dim=0)

                y_emb = encoder(Y[idx: idx + bsize])
                if y_mean is None:
                    y_mean = y_emb.sum(dim=0)
                else:
                    y_mean = y_mean + y_emb.sum(dim=0)

            x_mean = x_mean / float(num_samples)
            y_mean = y_mean / float(num_samples)

        self.x_mean = x_mean.detach()
        self.y_mean = y_mean.detach()

    def forward(self, x):

        return x + self.factor * (-self.x_mean + self.y_mean)


class MLP(Mapping):
    """
    This mapping is a standard (multi-layer) MLP.

    :param embedding_size: The dimensionality of the input (and output) embedding
    :type embedding_size: int
    :param n_layers: Number of layers (linear transform + activation function + dropout). 
        Does not include the output layer.
    :type n_layers: int
    :param hidden_size: Number of units in intermediate layers
    :type hidden_size: int
    :param nonlinearity: A PyTorch activation function such as ReLU or SELU
    :type nonlinearity: torch.nn.Module
    :param dropout: PyTorch module for performing dropout (e.g. `Dropout` or `AlphaDropout`)
    :type dropout: torch.nn.Module
    :param dropout_p: Dropout rate to apply to input at each layer. (0, 1)
    :type dropout_p: float
    """

    def __init__(self, embedding_size, n_layers, hidden_size, nonlinearity=nn.SELU,
                 dropout=nn.AlphaDropout, dropout_p=0):
        super(MLP, self).__init__(embedding_size)

        activation = nonlinearity
        self.dropout_p = dropout_p
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        layers = []
        activations = []
        dropouts = []
        for i in range(n_layers):
            if i == 0:
                layers.append(nn.Linear(embedding_size, hidden_size))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
            activations.append(activation())
            if dropout_p != 0:
                dropouts.append(dropout(p=dropout_p))

        layers.append(nn.Linear(hidden_size, embedding_size))
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.mlp = nn.ModuleList(layers)
        self.activations = nn.ModuleList(activations)
        self.dropouts = nn.ModuleList(dropouts)

    def forward(self, embeddings):
        x = embeddings

        for i in range(len(self.mlp)):
            x = self.mlp[i](x)

            # activations and dropout are NOT applied at output layer
            if i != len(self.mlp) - 1:
                x = self.activations[i](x)
                if self.dropout_p != 0:
                    x = self.dropouts[i](x)

        return x
