import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from .autoencoder import Encoder


class RNNEncoder(Encoder):
    def __init__(self, config):
        super(RNNEncoder, self).__init__(config)

        self.teacher_forcing_ratio = config.teacher_forcing_ratio
        self.gaussian_noise_std = config.gaussian_noise_std
        self.unit_sphere = config.unit_sphere
        self.teacher_forcing_batchwise = config.teacher_forcing_batchwise

        self.config = config
        self.device = config.device

        self.hidden_size = config.hidden_size
        self.variational = config.variational

        self.max_sequence_len = config.max_sequence_len
        self.input_size = config.hidden_size

        # plus 1 to make the 0th word a "padding" one.
        self.vocab_size = config.vocab_size + 1

        self.embedding = nn.Embedding(
            self.vocab_size, config.input_size, padding_idx=0)  # let 0 denote padding
        self.eos_idx = config.eos_idx
        self.sos_idx = config.sos_idx
        self.layers = config.layers
        self.reduction = "mean"

        self.type = config.type

        if self.type == "LSTM":
            self.encoder = nn.LSTM(
                input_size=config.input_size,
                hidden_size=self.hidden_size,
                num_layers=config.layers,
                bidirectional=True,
                batch_first=True
            )
        elif self.type == "GRU":
            self.encoder = nn.GRU(
                input_size=config.input_size,
                hidden_size=self.hidden_size,
                num_layers=config.layers,
                bidirectional=True,
                batch_first=True
            )

        if config.variational:
            self.hidden2mean = nn.Linear(self.hidden_size, self.hidden_size)
            self.hidden2logv = nn.Linear(self.hidden_size, self.hidden_size)

    def encode(self, x, lengths, train=False, reparameterize=True):
        embedded = self.embedding(x)

        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True)

        _, h = self.encoder(packed)

        # h_n of shape (num_layers*num_dir, batch, hidden_size)
        # mean over all hidden layers
        if self.type == "LSTM":
            h = h[0].mean(dim=0)
        else:
            h = h.mean(dim=0)

        if self.variational:
            mean = self.hidden2mean(h)
            logv = self.hidden2logv(h)
            std = torch.exp(0.5 * logv)
            if reparameterize:
                h = torch.randn(x.shape[0], self.hidden_size,
                                device=self.device) * std + mean
            else:
                h = mean

        if self.unit_sphere:
            h = h / h.norm(p=None, dim=-1, keepdim=True)

        # (batch, hidden_size)
        if train and self.variational:
            return h, mean, logv
        else:
            return h
