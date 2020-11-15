"""
Contains a list of all encoders.
"""
import torch
from torch import nn


class Encoder(nn.Module):
    """
    An encoder always takes a list of length 'b' as input and outputs a batch of
    embeddings of size 'b'.

    """

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.use_lookup = config['use_lookup']
        self.lookup_table = {}

    def lookup(self, S_batch):
        if self.use_lookup:
            existing = []
            non_existing = []
            for i, S in enumerate(S_batch):
                if S in self.lookup_table:
                    existing.append((i, self.lookup_table(S)))
                else:
                    non_existing.append((i, S))

            return existing, non_existing

        else:
            return [], zip(list(range(len(S_batch))), S_batch)

    def encode(self, S_list):
        """
        To be implemented. Takes a list of strings and returns a list of embeddings. 
        """
        pass

    def forward(self, S_batch):
        """
        Turns a list of strings into a list of embeddings. First checks if
        the embeddings have already been computed.
        """

        batch_size = len(S_batch)
        existing, non_existing = self.lookup(S_batch)

        ids, non_existing_S = zip(*non_existing)
        new_encoded = self.encode(non_existing_S)
        new_encoded = zip(ids, new_encoded)

        existing.extend(new_encoded)
        existing.sort(key=lambda x: x[0])

        _, embeddings = zip(*existing)
        embeddings = torch.cat(embeddings, dim=0).view(batch_size, -1)
        return embeddings


class Decoder(nn.Module):
    """
    A decoder takes a batch of embeddings of size 'b' as input and outputs a 
    batch of predictions (training time) or a list of texts (test time).

    """

    def __init__(self):
        super(Decoder, self).__init__()

    def predict(self, S_batch, target_batch=None):
        """
        To be implemented. Takes a batch of embeddings and returns a batch of
        predictions. At training time, target_batch contains a list of target
        sentences.
        """
        pass

    def prediction_to_text(self, predictions):
        """
        Takes a list of batch of embeddings of size b and returns a list of texts
        of length b.
        """
        pass

    def forward(self, embeddings, target_batch=None):
        """
        Turns a list of strings into a list of embeddings. First checks if
        the embeddings have already been computed.
        """
        outputs = self.predict(
            embeddings, target_batch=target_batch if self.training else None)
        if self.training:
            return outputs
        else:
            return self.prediction_to_text(outputs)
