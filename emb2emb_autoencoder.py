from torch.nn.utils.rnn import pad_sequence
from autoencoders.autoencoder import AutoEncoder
from autoencoders.rnn_encoder import RNNEncoder
from autoencoders.rnn_decoder import RNNDecoder
from emb2emb.encoding import Encoder, Decoder
from tokenizers import CharBPETokenizer, SentencePieceBPETokenizer
from emb2emb.utils import Namespace
import torch
import os
import json
import copy

HUGGINGFACE_TOKENIZERS = ["CharBPETokenizer", "SentencePieceBPETokenizer"]


def tokenize(s):
    # TODO: more sophisticated tokenization
    return s.split()


def get_tokenizer(tokenizer, location='bert-base-uncased'):
    # TODO: do we need to pass more options to the file?
    tok = eval(tokenizer)(vocab_file=location + '-vocab.json',
                          merges_file=location + '-merges.txt')
    tok.add_special_tokens(["[PAD]", "<unk>", "<SOS>", "<EOS>"])
    return tok


def get_autoencoder(config):
    if os.path.exists(config["default_config"]):
        with open(config["default_config"]) as f:
            model_config_dict = json.load(f)
    else:
        model_config_dict = {}
    with open(os.path.join(config["modeldir"], "config.json")) as f:
        orig_model_config = json.load(f)
        model_config_dict.update(orig_model_config)
        model_config = Namespace()
        model_config.__dict__.update(model_config_dict)

    tokenizer = get_tokenizer(
        model_config.tokenizer, model_config.tokenizer_location)
    model_config.__dict__["vocab_size"] = tokenizer.get_vocab_size()
    model_config.__dict__["sos_idx"] = tokenizer.token_to_id("<SOS>")
    model_config.__dict__["eos_idx"] = tokenizer.token_to_id("<EOS>")
    model_config.__dict__["unk_idx"] = tokenizer.token_to_id("<unk>")

    model_config.__dict__["device"] = config["device"]

    encoder_config, decoder_config = copy.deepcopy(
        model_config), copy.deepcopy(model_config)
    encoder_config.__dict__.update(model_config.__dict__[model_config.encoder])
    encoder_config.__dict__["tokenizer"] = tokenizer
    decoder_config.__dict__.update(model_config.__dict__[model_config.decoder])

    if model_config.encoder == "RNNEncoder":
        encoder = RNNEncoder(encoder_config)

    if model_config.decoder == "RNNDecoder":
        decoder = RNNDecoder(decoder_config)

    model = AutoEncoder(encoder, decoder, tokenizer, model_config)

    checkpoint = torch.load(os.path.join(
        config["modeldir"], model_config.model_file), map_location=config["device"])
    model.load_state_dict(checkpoint["model_state_dict"])

    return model


class AEEncoder(Encoder):
    def __init__(self, config):
        super(AEEncoder, self).__init__(config)
        self.device = config["device"]
        self.model = get_autoencoder(config)
        self.use_lookup = self.model.encoder.variational

    def _prepare_batch(self, indexed, lengths):
        X = pad_sequence([torch.tensor(index_list, device=self.device)
                          for index_list in indexed], batch_first=True, padding_value=0)
        lengths, idx = torch.sort(torch.tensor(
            lengths, device=self.device).long(), descending=True)
        return X[idx], lengths, idx

    def _undo_batch(self, encoded, sort_idx):
        ret = [[] for _ in range(encoded.shape[0])]
        for i, c in zip(sort_idx, range(encoded.shape[0])):
            ret[i] = encoded[c]
        return torch.stack(ret)

    def encode(self, S_list):

        indexed = [self.model.tokenizer.encode(
            "<SOS>" + s + "<EOS>").ids for s in S_list]

        lengths = [len(i) for i in indexed]
        X, X_lens, sort_idx = self._prepare_batch(indexed, lengths)
        encoded = self.model.encode(X, X_lens)
        # Since _prepare_batch sorts by length, we will need to undo this.
        return self._undo_batch(encoded, sort_idx)


class AEDecoder(Decoder):
    def __init__(self, config):
        super(AEDecoder, self).__init__()
        self.device = config["device"]
        self.model = get_autoencoder(config)

    def _prepare_batch(self, indexed, lengths):
        X = pad_sequence([torch.tensor(index_list, device=self.device)
                          for index_list in indexed], batch_first=True, padding_value=0)
        #lengths, idx = torch.sort(torch.tensor(lengths, device=self.device).long(), descending=True)
        # return X[idx], lengths, idx
        lengths = torch.tensor(lengths, device=self.device).long()
        return X, lengths

    def _encode(self, S_list):

        indexed = [self.model.tokenizer.encode(
            "<SOS>" + s + "<EOS>").ids for s in S_list]

        lengths = [len(i) for i in indexed]
        X, X_lens = self._prepare_batch(indexed, lengths)
        return X, X_lens

    def predict(self, S_batch, target_batch=None):
        if self.training:
            target_batch, target_length = self._encode(target_batch)
            out = self.model.decode_training(
                S_batch, target_batch, target_length)
            return out, target_batch
        else:
            return self.model.decode(S_batch, beam_width=15)

    def prediction_to_text(self, predictions):
        predictions = [self.model.tokenizer.decode(
            p, skip_special_tokens=True) for p in predictions]
        return predictions
