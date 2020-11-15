"""
Module to train mapping and the baseline model.
"""
import torch
from torch import nn
from random import choices
from .fgim import fast_gradient_iterative_modification
import time
from emb2emb.fgim import make_binary_classification_loss

MODE_EMB2EMB = "mapping"
MODE_SEQ2SEQ = "seq2seq"
MODE_FINETUNEDECODER = "finetune_decoder"
MODE_SEQ2SEQFREEZE = "seq2seq_freeze"


class Emb2Emb(nn.Module):
    """This class encapsulates the computations happening in the Task-Learning phase of the Emb2Emb framework during
    training and inference.

    The basic flow in the Emb2Emb framework is like this:

    #. Train an autoencoder to receive and encoder and a decoder.
    #. Freeze the encoder and decoder.
    #. Train a mapping in the autoencoder embedding space that maps the 
        encoding of the input to the encoding of the (desired) output.
    #. At inference time, encode the input, plug it into the mapping, (optionally)
        apply Fast-Gradient-Iterative-Modification, and plug the result into the decoder.

    This class implements steps 2, 3, and 4. Emb2Emb can be used with any pretrained
    autoencoder, so the encoder and decoder are passed for initialization.
    Moreover, initialization expects the specific mapping (architecture) and loss function to be used
    for training.

    Learning in the embedding space has the disadvantage that the produced outputs
    are not necessarily such that the decoder can deal with them. To mitigate this
    issue, training in Emb2Emb uses an optional adversarial loss term that encourages
    the mapping to keep its outputs on the manifold of the autoencoder such that the
    decoder can more likely handle them well. 

    :param encoder: Used for encoding the input and, if provided, the output sequence.
    :type encoder: class:`mapping.encoding.Encoder`
    :param decoder: Used for decoding the output of the mapping. 
    :type decoder: class:`mapping.encoding.Decoder`
    :param mapping: Used for transforming the embedding of the input to the embedding of the output.
    :type mapping: class:`emb2emb.mapping.Mapping`
    :param loss_fn: A loss function for regression problems, i.e., it must take as input
        a pair (predicted, true) of embedding tensors of shape [batch size, embedding_dim].
    :type loss_fn: class:`torch.nn.Module`
    :param mode: 
    :type mode:
    :param use_adversarial_term: If set, adversarial regularization term will be used.
    :param adversarial_lambda Weight of the adversarial loss term.
    :param device: The device to initialize tensors on.
    :param critic_lr: Learning rate for training the discriminator/critic.
    :param embedding_dim: Dimensionality of fixed-size bottleneck embedding.
    :param critic_hidden_units: Hidden units in the discriminator MLP.
    :param critic_hidden_layers: Number of hidden layers in the discriminator MLP.
    :param real_data: If set to "input", the discriminator will receive target sequences
        as the "true" embeddings. Otherwise, the parameter will be interpreted as a path
        to file containing a corpus, with one sentence per line. Positive examples for
        the descriminator are then randomly chosen from that corpus.
    :param fast_gradient_iterative_modification: Whether to use FGIM at inference time.
    :param binary_classifier: The binary classifier that FGIM takes the derivative of with
        respect to the input. The gradient is followed towards the classifying the input as
        '1'.
    :param fgim_decay: Rate by which to decay weights.
    :param fgim_threshold: How far from the target '1' the target has to be in order
        to be considered finished.
    :param fgim_weights: Step sizes to be applied in parallel (list).
    :param fgim_use_training_loss: If set to true, the training loss is also followed
        at inference time (i.e., including the adversarial term if active).
    :param fgim_start_at_y: Instead of computing FGIM gradients starting from the output
        of the mapping, we start from the embedding of the target (which is the same as
        the input in the unsupervised case).
    """

    def __init__(self, encoder, decoder, mapping, loss_fn, mode,
                 use_adversarial_term=False,
                 adversarial_lambda=0.,
                 device=None,
                 critic_lr=0.001,
                 embedding_dim=512,
                 critic_hidden_units=512,
                 critic_hidden_layers=1,
                 real_data="input",
                 fast_gradient_iterative_modification=False,
                 binary_classifier=None,
                 fgim_decay=1.0,
                 fgim_weights=[10e0, 10e1, 10e2, 10e3],
                 fgim_loss_f=None,
                 fgim_criterion_f=None,
                 fgim_start_at_y=False,
                 fgim_max_steps=30):
        """Constructor method"""
        super(Emb2Emb, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mapping = mapping
        self.loss_fn = loss_fn
        self.fgim_decay = fgim_decay
        self.fgim_loss_f = fgim_loss_f
        self.fgim_criterion_f = fgim_criterion_f
        self.fgim_start_at_y = fgim_start_at_y
        self.fgim_max_steps = fgim_max_steps
        self.fgim_weights = fgim_weights
        self.change_mode(mode)
        self.track_input_output_distance = False
        self.use_adversarial_term = use_adversarial_term
        self.fast_gradient_iterative_modification = fast_gradient_iterative_modification
        self.binary_classifier = binary_classifier
        self.total_time_fgim = 0.
        self.total_emb2emb_time = 0.
        self.total_inference_time = 0.

        if mode in [MODE_FINETUNEDECODER, MODE_EMB2EMB, MODE_SEQ2SEQFREEZE]:
            for p in self.encoder.parameters():
                p.requires_grad = False

        if mode in [MODE_FINETUNEDECODER]:
            for p in self.mapping.parameters():
                p.requires_grad = False

        if mode in [MODE_EMB2EMB, MODE_SEQ2SEQFREEZE]:
            for p in self.decoder.parameters():
                p.requires_grad = False

        if use_adversarial_term:
            hidden = critic_hidden_units
            critic_layers = [nn.Linear(embedding_dim,
                                       hidden),
                             nn.ReLU()]
            for _ in range(critic_hidden_layers):
                critic_layers.append(nn.Linear(hidden, hidden))
                critic_layers.append(nn.ReLU())
            critic_layers.append(nn.Linear(hidden, 2))
            self.critic = [nn.Sequential(*critic_layers)]
            self.real_data = real_data
            self.critic_loss = nn.CrossEntropyLoss()
            self.critic_optimizer = torch.optim.Adam(
                self._get_critic().parameters(), lr=critic_lr)
            dev = device
            self._get_critic().to(dev)
            self.critic_loss.to(dev)

            self.adversarial_lambda = adversarial_lambda

    def _get_critic(self):
        return self.critic[0]

    def change_mode(self, new_mode):
        if not new_mode in [MODE_EMB2EMB, MODE_SEQ2SEQ, MODE_FINETUNEDECODER, MODE_SEQ2SEQFREEZE]:
            raise ValueError("Invalid mode.")
        self.mode = new_mode

    def _decode(self, output_embeddings, target_batch=None, Y_embeddings=None):
        if self.mode == MODE_EMB2EMB or not self.training:

            if self.fast_gradient_iterative_modification:
                # Fast Gradient Iterative Modification
                start_time = time.time()

                # configure FGIM
                if self.fgim_start_at_y:
                    starting_point = Y_embeddings
                else:
                    starting_point = output_embeddings

                output_embeddings = fast_gradient_iterative_modification(
                    starting_point, lambda x: self.fgim_loss_f(
                        x, Y_embeddings), self.fgim_criterion_f,
                    self.fgim_weights, self.fgim_decay, self.fgim_max_steps)

                self.total_time_fgim = self.total_time_fgim + \
                    (time.time() - start_time)

            outputs = self.decoder(output_embeddings)
            return outputs

        elif self.mode in [MODE_SEQ2SEQ, MODE_FINETUNEDECODER, MODE_SEQ2SEQFREEZE]:
            outputs, targets = self.decoder(
                output_embeddings, target_batch=target_batch)
            vocab_size = outputs.size(-1)
            outputs = outputs.view(-1, vocab_size)
            targets = targets.view(-1)
            return outputs, targets
        else:
            raise ValueError(
                "Undefined behavior for encoding in mode " + self.mode)

    def _encode(self, S_batch):
        if self.mode in [MODE_EMB2EMB, MODE_FINETUNEDECODER, MODE_SEQ2SEQ, MODE_SEQ2SEQFREEZE]:
            embeddings = self.encoder(S_batch)
        else:
            raise ValueError(
                "Undefined behavior for encoding in mode " + self.mode)
        return embeddings

    def _train_critic(self, real_embeddings, generated_embeddings):
        self._get_critic().train()

        # need to detach from the current computation graph, because critic has
        # its own computation graph
        real_embeddings = real_embeddings.detach().clone()
        generated_embeddings = generated_embeddings.detach().clone()

        # get predictions from critic
        all_embeddings = torch.cat(
            [real_embeddings, generated_embeddings], dim=0)
        critic_logits = self._get_critic()(all_embeddings)

        # compute critic loss
        true_labels = torch.ones(
            (real_embeddings.shape[0]), device=real_embeddings.device, dtype=torch.long)
        false_labels = torch.zeros(
            (generated_embeddings.shape[0]), device=generated_embeddings.device, dtype=torch.long)
        labels = torch.cat([true_labels, false_labels], dim=0)
        loss = self.critic_loss(critic_logits, labels)

        # train critic
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return loss

    def _test_critic(self, embeddings):
        self._get_critic().eval()

        # with torch.no_grad():
        # do not detach embeddings, because we need to propagate through critic
        # within the same computation graph
        critic_logits = self._get_critic()(embeddings)

        labels = torch.zeros(
            (embeddings.shape[0]), device=embeddings.device, dtype=torch.long)
        loss = self.critic_loss(critic_logits, labels)
        return loss

    def _adversarial_training(self, loss, output_embeddings, Y_embeddings):

        # train the discriminator
        # NOTE: We need to train the discriminator first, because otherwise we would
        # backpropagate through the critic after changing it
        train_critic_loss = self._train_critic(
            Y_embeddings, output_embeddings)

        task_loss = loss.clone()

        # what does the discriminator say about the predicted output
        # embeddings?
        critic_loss = self._test_critic(output_embeddings)

        # we want to fool the critic, i.e., we want to its loss to be high =>
        # subtract adversarial loss
        loss = loss - self.adversarial_lambda * critic_loss

        return loss, task_loss, critic_loss, train_critic_loss

    def compute_emb2emb(self, Sx_batch):
        # encode input
        X_embeddings = self._encode(Sx_batch)

        # mapping step
        if not self.training:  # measure the time it takes to run through mapping, but only at inference time
            s_time = time.time()

        output_embeddings = self.mapping(X_embeddings)

        if not self.training:
            self.total_emb2emb_time = self.total_emb2emb_time + \
                (time.time() - s_time)

        return output_embeddings, X_embeddings

    def compute_loss(self, output_embeddings, Y_embeddings):
        loss = self.loss_fn(output_embeddings, Y_embeddings)

        if self.use_adversarial_term:

            if self.real_data == "input":
                real_data = Y_embeddings
            else:
                # unless we're using the true output embeddings at
                real_data = self._encode(
                    choices(self.real_data, k=Y_embeddings.size(0)))

            return self._adversarial_training(loss, output_embeddings, real_data)
        return loss

    def forward(self, Sx_batch, Sy_batch):
        """
        Propagates through the mapping framework. Takes as input two lists of
        texts corresponding to the input and outputs. Returns loss (single scalar)
        if in training mode, otherwise returns texts.
        """
        # measure inference time it takes
        if not self.training:
            s_time = time.time()

        output_embeddings, X_embeddings = self.compute_emb2emb(Sx_batch)

        if self.training:
            # compute loss depending on the mode

            if self.mode == MODE_EMB2EMB:
                Y_embeddings = self._encode(Sy_batch)

                loss = self.compute_loss(output_embeddings, Y_embeddings)
                if self.use_adversarial_term:
                    loss, task_loss, critic_loss, train_critic_loss = loss

                if self.track_input_output_distance:
                    input_output_distance = self.loss_fn(
                        X_embeddings, Y_embeddings)
                    print(input_output_distance)
            elif self.mode in [MODE_SEQ2SEQ, MODE_FINETUNEDECODER, MODE_SEQ2SEQFREEZE]:
                # for training with CE
                outputs, targets = self._decode(
                    output_embeddings, target_batch=Sy_batch)
                loss = self.loss_fn(outputs, targets)

            if self.use_adversarial_term:
                return loss, task_loss, critic_loss, train_critic_loss
            else:
                return loss
        else:
            # return textual output
            out = self._decode(output_embeddings, Y_embeddings=X_embeddings)
            self.total_inference_time = self.total_inference_time + \
                (time.time() - s_time)
            return out
