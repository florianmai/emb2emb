from torch import nn
import torch
from random import shuffle
from torch.nn.modules.loss import BCEWithLogitsLoss
import numpy as np
import os


class BinaryClassifier(nn.Module):
    """
    """

    def __init__(self, input_size, hidden_size, dropout=0., gaussian_noise_std=0.):
        super(BinaryClassifier, self).__init__()

        self.classifier = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(input_size, hidden_size),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(hidden_size, 1))

        self.gaussian_noise_std = gaussian_noise_std

    def forward(self, inputs):

        if self.training and self.gaussian_noise_std > 0.:
            inputs = inputs + \
                torch.randn_like(inputs) * self.gaussian_noise_std

        return self.classifier(inputs)


def freeze(m):
    for p in m.parameters():
        p.requires_grad = False


def train_binary_classifier(true_inputs, false_inputs, encoder, params, num_val_samples=1000):

    outputmodelname = params.outputmodelname + "_binary_clf"
    if params.load_binary_clf:
        binary_classifier = BinaryClassifier(
            params.embedding_dim, 512, 0., 0.).to(encoder.device)
        checkpoint = torch.load(os.path.join(params.outputdir, outputmodelname),
                                map_location=params.device)
        binary_classifier.load_state_dict(checkpoint["model_state_dict"])
        return binary_classifier

    inputs = true_inputs + false_inputs
    t = ([1] * len(true_inputs)) + ([0] * len(false_inputs))

    # get validation set
    indices = list(range(len(inputs)))
    inputs, t = np.array(inputs), np.array(t)
    shuffle(indices)
    val_inputs = inputs[indices[-num_val_samples:]]
    val_targets = t[indices[-num_val_samples:]]
    inputs = inputs[indices[:-num_val_samples]]
    t = t[indices[:-num_val_samples]]
    indices = list(range(len(inputs)))

    binary_classifier = BinaryClassifier(params.embedding_dim,
                                         512,
                                         params.dropout_binary,
                                         params.gaussian_noise_binary).to(encoder.device)
    opt = torch.optim.Adam(binary_classifier.parameters(), lr=params.lr_bclf)
    freeze(encoder)
    encoder.eval()
    loss_f = BCEWithLogitsLoss()

    def save_clf():
        checkpoint = {"model_state_dict": binary_classifier.state_dict()}
        torch.save(checkpoint, os.path.join(params.outputdir, outputmodelname))

    best_acc = evaluate(val_inputs, val_targets, encoder,
                        binary_classifier, params)
    bsize = params.batch_size
    correct = 0.
    for e in range(params.n_epochs_binary):

        # shuffle data in each epoch
        shuffle(indices)
        inputs = inputs[indices]
        t = t[indices]

        binary_classifier.train()
        losses = []
        for idx in range(0, len(inputs), bsize):
            ib = inputs[idx: idx + bsize]
            tb = t[idx: idx + bsize]

            tb = torch.tensor(tb, device=encoder.device).view(-1, 1).float()
            with torch.no_grad():
                embeddings = encoder(ib)
            preds = binary_classifier(embeddings)
            acc = ((preds > 0.5) == tb).sum()
            loss = loss_f(preds, tb)
            correct += acc

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

            if (idx / bsize) % params.log_freq == 0:
                avg_loss = np.array(losses[-params.log_freq:]).mean()
                print("Binary classification step {}<->{}: loss {} ; t-acc: {}, v-acc: {}".format(e,
                                                                                                  idx,
                                                                                                  avg_loss,
                                                                                                  correct /
                                                                                                  float(
                                                                                                      params.log_freq * bsize),
                                                                                                  best_acc))
                correct = 0.

        val_acc = evaluate(val_inputs, val_targets, encoder,
                           binary_classifier, params)
        if val_acc > best_acc:
            best_acc = val_acc
            save_clf()
        print("Loss in epoch {}: {}".format(e, np.array(losses).mean()))

    return binary_classifier


def evaluate(val_inputs, val_targets, encoder, binary_classifier, params):
    inputs = val_inputs
    t = val_targets
    bsize = params.batch_size

    correct = 0.
    binary_classifier.eval()

    for idx in range(0, len(inputs), bsize):
        ib = inputs[idx: idx + bsize]
        tb = t[idx: idx + bsize]

        tb = torch.tensor(tb, device=encoder.device).view(-1, 1).float()
        with torch.no_grad():
            embeddings = encoder(ib)
        preds = binary_classifier(embeddings)
        acc = ((preds > 0.5) == tb).sum()
        correct += acc

    return float(correct) / len(inputs)
