import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
from .autoencoder import Decoder


class RNNDecoder(Decoder):
    def __init__(self, config):
        super(RNNDecoder, self).__init__(config)

        self.teacher_forcing_ratio = config.teacher_forcing_ratio
        self.unit_sphere = config.unit_sphere
        self.teacher_forcing_batchwise = config.teacher_forcing_batchwise

        self.config = config
        self.device = config.device

        self.vocab_size = config.vocab_size + 1

        self.type = config.type

        self.hidden_size = config.hidden_size

        self.max_sequence_len = config.max_sequence_len
        self.input_size = config.hidden_size

        self.embedding = nn.Embedding(
            self.vocab_size, config.input_size, padding_idx=0)  # let 0 denote padding
        self.eos_idx = config.eos_idx
        self.sos_idx = config.sos_idx
        self.unk_idx = config.unk_idx

        self.word_dropout = config.word_dropout
        self.layers = config.layers

        # Consider using GRU?
        if self.type == "LSTM":
            self.decoder = nn.LSTM(
                input_size=config.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.layers,
                batch_first=True
            )
        elif self.type == "GRU":
            self.decoder = nn.GRU(
                input_size=config.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.layers,
                batch_first=True
            )

        # let 0 denote padding
        self.out = nn.Linear(self.hidden_size, config.vocab_size + 1)

    def init_hidden(self, x):
        if self.type == "LSTM":
            return x.repeat(self.layers, 1, 1), torch.zeros(self.layers, x.shape[0], self.hidden_size, device=self.device)
        elif self.type == "GRU":
            return x.repeat(self.layers, 1, 1)

    def decode(self, x, train=False, actual=None, lengths=None, beam_width=1):
        if self.unit_sphere:
            h = h / h.norm(p=None, dim=-1, keepdim=True)

        if not train:
            if beam_width != 1:
                return self.beam_decode(x, beam_width)
            else:
                return self.greedy_decode(x)
        else:
            h = self.init_hidden(x)

            embedded_input = self.embedding(torch.tensor(
                [[self.sos_idx]], device=self.device).repeat(x.shape[0], 1))

            predictions = []

            for t in range(1, lengths.max()):
                output, h = self.decoder(embedded_input, h)
                # lstm input: (batch, seq_len, input_size)
                # lstm output: (batch, seq_len, hidden_size)
                res = self.out(output.squeeze(1))

                ret = res.clone()
                ret *= torch.gt(lengths.reshape(-1, 1), t).float()

                predictions.append(ret)

                if random.random() < self.teacher_forcing_ratio:
                    next_token = actual[:, t].reshape(-1, 1)
                else:
                    topv, topi = res.topk(1)
                    next_token = topi.detach()
                if train and random.random() < self.word_dropout:
                    next_token = torch.tensor(
                        [[self.unk_idx]], device=self.device).repeat(x.shape[0], 1)

                embedded_input = self.embedding(next_token)

            predictions = torch.stack(predictions).permute(1, 0, 2)
            # is: seq, batch, pred
            # want: batch, seq, pred

            # Add SOS prediction to the output
            sos_padding = torch.zeros(
                (x.shape[0], 1, self.vocab_size), device=self.device)
            sos_padding[:, :, self.sos_idx] = 1
            return torch.cat((sos_padding, predictions), 1)

    def decode_teacher_forcing(self, x, actual, lengths):
        h = self.init_hidden(x)

        # We want to feed everything but the last element (so the network can
        # predict the <EOS> token). We copy the actual sequence, remove <EOS>
        # token, then reshape the seq_len.
        teacher_input = actual.clone()
        teacher_input[torch.arange(
            teacher_input.shape[0], device=self.device), lengths - 1] = 0
        if self.train and self.word_dropout > 0.:
            mask = torch.rand_like(
                teacher_input, device=teacher_input.device) < self.word_dropout
            teacher_input[mask] = self.unk_idx
        embedded_teacher = self.embedding(
            teacher_input[:, :teacher_input.shape[1] - 1])

        packed_teacher = pack_padded_sequence(
            embedded_teacher, lengths - 1, batch_first=True)

        packed_output, h = self.decoder(packed_teacher, h)

        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # A "hacky" way to run the dense layer per timestep
        predictions = self.out(
            output.contiguous().view(
                -1, output.shape[2])).reshape(
                    output.shape[0], output.shape[1], self.vocab_size)

        # Add SOS prediction to the output
        sos_padding = torch.zeros(
            (x.shape[0], 1, self.vocab_size), device=self.device)
        sos_padding[:, :, self.sos_idx] = 1

        return torch.cat((sos_padding, predictions), 1)
        # return self.softmax(predictions)  # Commented since cross entropy
        # does a softmax

    def decode_train_greedy(self, x, lengths):
        h = self.init_hidden(x)

        embedded_input = self.embedding(torch.tensor(
            [[self.sos_idx]], device=self.device).repeat(x.shape[0], 1))

        predictions = []

        for t in range(1, lengths.max()):
            output, h = self.decoder(embedded_input, h)
            # lstm input: (batch, seq_len, input_size)
            # lstm output: (batch, seq_len, hidden_size)
            res = self.out(output.squeeze(1))

            ret = res.clone()
            ret *= torch.gt(lengths.reshape(-1, 1), t).float()

            predictions.append(ret)

            topv, topi = res.topk(1)

            embedded_input = self.embedding(topi.detach())

        predictions = torch.stack(predictions).permute(1, 0, 2)
        # is: seq, batch, pred
        # want: batch, seq, pred

        # Add SOS prediction to the output
        sos_padding = torch.zeros(
            (x.shape[0], 1, self.vocab_size), device=self.device)
        sos_padding[:, :, self.sos_idx] = 1
        return torch.cat((sos_padding, predictions), 1)

    # Removes the extra EOS tokens added
    def clip_predictions(self, pred):
        results = []
        for s in pred:
            curr = []
            for idx in s:
                curr.append(idx)
                if idx == self.eos_idx:
                    break
            results.append(curr)
        return results

    class BeamNode:
        def __init__(self, hidden_state, previous_node, word_id, log_prob, length):
            self.hidden_state = hidden_state
            self.previous_node = previous_node
            self.word_id = word_id
            self.log_prob = log_prob
            self.length = length

    # Greedy decode for LSTMAE and LSTMAE
    def greedy_decode(self, x):
        h = self.init_hidden(x)

        embedded_input = self.embedding(torch.tensor(
            [[self.sos_idx]], device=self.device).repeat(x.shape[0], 1))

        predictions = [[self.sos_idx] for _ in range(x.shape[0])]

        for t in range(1, self.max_sequence_len):
            output, h = self.decoder(embedded_input, h)

            res = self.out(output.squeeze(1))
            topv, topi = res.topk(1)

            done_count = 0
            for b in range(x.shape[0]):
                if predictions[b][-1] != self.eos_idx:
                    predictions[b].append(topi[b].cpu().item())

                    # if last token placed, and not eos, just cut off
                    if t == self.max_sequence_len - 1 and predictions[b][-1] != self.eos_idx:
                        predictions[b].append(self.eos_idx)
                else:
                    done_count += 1
            if done_count == x.shape[0]:
                break

            embedded_input = self.embedding(topi.detach())

        return self.clip_predictions(predictions)

    # Only works for LSTM
    def beam_decode(self, x, beam_width=10):
        # x = (batch, hidden_size)
        # hidden_lstm = (layers, batch, hidden)
        h = self.init_hidden(x)
        decoded = [None for i in range(x.shape[0])]

        # beam_width nodes per batch
        incomplete = {ba: [
            self.BeamNode(h, None, torch.tensor(self.sos_idx, device=self.device), 0, 1) for be in range(beam_width)
        ] for ba in range(x.shape[0])}

        # create first hypotheses:
        # lstm input: (batch, seq_len, input_size)
        # lstm output: (batch, seq_len, hidden_size)
        embedded_input = self.embedding(torch.tensor(
            [[self.sos_idx]], device=self.device).repeat(x.shape[0], 1))
        decoder_output, h = self.decoder(embedded_input, h)

        # h_n of shape (num_layers, batch, hidden_size)
        for b in range(x.shape[0]):
            # decoder_output[b] shape: (1, hidden_size)
            log_probs = F.log_softmax(
                self.out(decoder_output[b]), dim=1).squeeze(0)
            k_log_probs, k_indices = torch.topk(log_probs, beam_width)
            for i in range(beam_width):
                prev_node = incomplete[b][i]
                if self.type == "LSTM":
                    incomplete[b][i] = self.BeamNode(
                        (h[0][:, b], h[1][:, b]), prev_node, k_indices[i], k_log_probs[i], 2)
                elif self.type == "GRU":
                    incomplete[b][i] = self.BeamNode(
                        h[:, b], prev_node, k_indices[i], k_log_probs[i], 2)

        for t in range(2, self.max_sequence_len):
            if len(incomplete) == 0:
                break
            # Prepare step [ batch1_beams | batch2_beams | | ]
            embedding_input = torch.tensor(
                [beam.word_id for batch in incomplete for beam in incomplete[batch]], device=self.device)
            # keep track of the order which beams are put in
            input_order = [batch for batch in incomplete]
            # embedding_input shape: (batch * beam_len)
            embedding_input = embedding_input.reshape(-1, 1)
            # embedding_input shape: (batch*beam_len, 1[seq_len])
            embedded_input = self.embedding(embedding_input)
            # embedded_input shape: (batch*beam_len, 1, input_size)

            # want: h_prev of (num_layers, batch*beam_len, input_size)
            # Do (batch*beam_len, num_layers, input_size) then move axis
            if self.type == "LSTM":
                h_prev = torch.stack(
                    [beam.hidden_state[0] for batch in incomplete for beam in incomplete[batch]]).permute(1, 0, 2)
                c_prev = torch.stack(
                    [beam.hidden_state[1] for batch in incomplete for beam in incomplete[batch]]).permute(1, 0, 2)
                h = (h_prev.contiguous(), c_prev.contiguous())
            elif self.type == "GRU":
                h = torch.stack(
                    [beam.hidden_state for batch in incomplete for beam in incomplete[batch]]).permute(1, 0, 2).contiguous()

            decoder_output, h = self.decoder(embedded_input, h)
            # lstm output: (batch*beam_len, 1, hidden_size)
            for batch_index, batch in enumerate(input_order):
                # Each batch is a seperate beam search.
                # Get the probabilites from each beam
                log_probs = F.log_softmax(self.out(
                    decoder_output[batch_index * beam_width:(batch_index + 1) * beam_width].squeeze(1)), dim=1)

                # Put all the beam probabilities in a single vector, with the
                # full seq prob
                seq_probs = torch.cat(
                    [incomplete[batch][i].log_prob + log_probs[i] for i in range(beam_width)])

                # Get the top k
                k_seq_probs, k_indices = torch.topk(seq_probs, beam_width)

                new_beams = []

                for seq_prob, index in zip(k_seq_probs, k_indices):
                    beam_index = index // self.vocab_size
                    word_index = index % self.vocab_size
                    prev_beam = incomplete[batch][beam_index]
                    if word_index == self.eos_idx:
                        # we hit the end of the sequence! Therefore, this element
                        # of the batch is now complete.

                        # Since we wont be training, we will turn these into regular
                        # values, rather than tensors.
                        seq = [self.eos_idx]
                        prev = prev_beam
                        while prev != None:
                            seq.append(prev.word_id.cpu().item())
                            prev = prev.previous_node
                        seq = seq[::-1]
                        decoded[batch] = seq
                        del incomplete[batch]
                        break
                    if self.type == "LSTM":
                        new_beams.append(
                            self.BeamNode(
                                (h[0][:, batch_index * beam_width + beam_index],
                                 h[1][:, batch_index * beam_width + beam_index]),
                                prev_beam,
                                word_index,
                                seq_prob,
                                prev_beam.length + 1))
                    elif self.type == "GRU":
                        new_beams.append(
                            self.BeamNode(
                                h[:, batch_index * beam_width + beam_index],
                                prev_beam,
                                word_index,
                                seq_prob,
                                prev_beam.length + 1))

                # if we didn't complete the sequence
                if batch in incomplete:
                    incomplete[batch] = new_beams

        # For elements which hit the max seq length, we will cut them off at the
        # most probable sequence so far.
        for batch in incomplete:
            seq = [self.eos_idx]
            # The first beam will be the most probable sequence so far
            prev = incomplete[batch][0]
            while prev != None:
                seq.append(prev.word_id.cpu().item())
                prev = prev.previous_node
            seq = seq[::-1]
            decoded[batch] = seq

        return self.clip_predictions(decoded)
