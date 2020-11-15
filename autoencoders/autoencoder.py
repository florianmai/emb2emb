import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from .noise import noisy


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config

    def encode(self, x, lengths, train=False):
        pass


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config

    def decode(self, x, train=False, actual=None, lengths=None, beam_width=1):
        pass

    def decode_teacher_forcing(self, x, actual, lengths):
        pass


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, tokenizer, config):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.config = config
        self.adversarial = config.adversarial
        self.variational = config.variational
        self.denoising = config.denoising
        self.rae_regularization = config.rae_regularization
        if self.config.share_embedding:
            self.decoder.embedding = self.encoder.embedding
        if self.adversarial:
            self.discriminator = nn.Sequential(
                torch.nn.Linear(config.hidden_size,
                                config.hidden_size, bias=False),
                torch.nn.SELU(),
                torch.nn.Linear(config.hidden_size,
                                config.hidden_size, bias=False),
                torch.nn.SELU(),
                torch.nn.Linear(config.hidden_size, 1, bias=True),
                torch.nn.Sigmoid()
            )
            self.optimD = torch.optim.Adam(
                self.discriminator.parameters(), lr=config.discriminator_lr)

    def forward(self, x, lengths):
        # denoising
        if self.training and self.denoising:
            x_, lengths_, orig_indices = noisy(
                self.tokenizer, x, self.config.p_drop)
        else:
            x_, lengths_ = x, lengths

        # x shape:                  (batch, seq_len)
        encoded = self.encoder.encode(x_, lengths_, train=True)

        if self.training and self.denoising:
            encoded = encoded[orig_indices]

        if self.variational:
            encoded, mean, logv = encoded
        # encoded shape:            (batch, hidden_size)

        # add small gaussian noise during training
        if self.training and self.config.gaussian_noise_std > 0.:
            encoded = encoded + \
                torch.randn_like(encoded) * self.config.gaussian_noise_std

        if self.config.teacher_forcing_batchwise and self.config.teacher_forcing_ratio > random.random():
            decoded_pred = self.decoder.decode_teacher_forcing(
                encoded, x, lengths)
        else:
            decoded_pred = self.decoder.decode(
                encoded, train=True, actual=x, lengths=lengths)

        # ret:                      (batch, seq_len, classes)
        if self.variational:
            return decoded_pred, mean, logv, encoded
        if self.adversarial:
            # it's important to detach the encoded embedding before feeding into the
            # discriminator so that when updating the discriminator, it doesn't
            # backprop through the generator
            encoded_det = encoded.detach().clone()
            prior_data = torch.randn_like(encoded)
            return decoded_pred, self.discriminator(encoded), self.discriminator(encoded_det), self.discriminator(prior_data), encoded
        else:
            return decoded_pred, encoded

    def encode(self, x, lengths):
        return self.encoder.encode(x, lengths, reparameterize=False)

    def decode(self, x, beam_width=1):
        return self.decoder.decode(x, beam_width=beam_width)

    def decode_training(self, h, actual, lengths):
        """
        Decoding step to be used for downstream training
        """
        return self.decoder.decode(h, train=True, actual=actual, lengths=lengths)

    def loss(self, predictions, embeddings, labels, reduction="mean"):
        # predictions:  (batch, seq_len, classes)
        # labels:       (batch, seq_len)

        l_rec = F.cross_entropy(
            predictions.reshape(-1, predictions.shape[2]), labels.reshape(-1), ignore_index=0, reduction=reduction)

        # regularize embeddings
        if self.rae_regularization > 0.:
            l_reg = ((embeddings.norm(dim=-1) ** 2) / 2.).mean()
            l = l_reg * self.rae_regularization + l_rec
            return l
        else:
            return l_rec

    def loss_variational(self, predictions, embeddings, labels, mu, z_var, lambda_r=1, lambda_kl=1, reduction="mean"):
        recon_loss = self.loss(predictions, embeddings,
                               labels, reduction=reduction)
        raw_kl_loss = torch.exp(z_var) + mu**2 - 1.0 - z_var
        if reduction == "mean":
            kl_loss = 0.5 * torch.mean(raw_kl_loss)
        elif reduction == "sum":
            kl_loss = 0.5 * torch.sum(raw_kl_loss)
        return lambda_r * recon_loss + lambda_kl * kl_loss, recon_loss, kl_loss

    def loss_adversarial(self, predictions, embeddings, labels, fake_z_g, fake_z_d, true_z, lambda_a=1):
        r_loss = self.loss(predictions, embeddings, labels)
        d_loss = (F.binary_cross_entropy(true_z, torch.ones_like(true_z)) +
                  F.binary_cross_entropy(fake_z_d, torch.zeros_like(fake_z_d))) / 2
        g_loss = F.binary_cross_entropy(fake_z_g, torch.ones_like(fake_z_g))
        # we need to update discriminator and generator independently, otherwise
        # we will update the generator to produce better distinguishable embeddings,
        # which we do not want
        return (r_loss + lambda_a * g_loss), r_loss, d_loss, g_loss

    def eval(self, x, lengths, teacher_forcing=False, beam_width=1):
        encoded = self.encoder.encode(x, lengths)
        # encoded shape:            (batch, hidden_size)
        if teacher_forcing:
            return self.decoder.decode_teacher_forcing(encoded, x, lengths)
        else:
            return self.decoder.decode(encoded, beam_width=beam_width)
