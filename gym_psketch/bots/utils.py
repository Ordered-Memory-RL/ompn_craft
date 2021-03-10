import torch
from torch import nn as nn

from gym_psketch import ENV_EMB, env_list, ID2SKETCHLEN


class SketchEmbedding(nn.Module):
    """ An env emb from hard coding """
    def __init__(self, emb_size):
        super(SketchEmbedding, self).__init__()
        self.env2sketchs = nn.Parameter(data=torch.tensor(ENV_EMB).long(), requires_grad=False)
        self.sketch_embedding = nn.Embedding(num_embeddings=self.env2sketchs.max().item() + 1,
                                             embedding_dim=emb_size)
        self.emb_size = emb_size

    def forward(self, env_ids):
        sketchs = nn.functional.embedding(env_ids, self.env2sketchs)
        sketchs_emb = self.sketch_embedding(sketchs)
        return sketchs_emb.sum(-2)


class GRUSketchEmbedding(nn.Module):
    def __init__(self, emb_size):
        super(GRUSketchEmbedding, self).__init__()
        self.env2sketchs = nn.Parameter(data=torch.tensor(ENV_EMB).long(), requires_grad=False)
        self.sketch_embedding = nn.Embedding(num_embeddings=self.env2sketchs.max().item() + 1,
                                             embedding_dim=emb_size)
        self.sketch_lengths = nn.Parameter(torch.tensor(ID2SKETCHLEN).long(), requires_grad=False)
        self.emb_size = emb_size
        self.gru = torch.nn.GRU(input_size=emb_size, hidden_size=emb_size, batch_first=True)

    def forward(self, env_ids):
        sketchs = nn.functional.embedding(env_ids, self.env2sketchs)
        sketch_lengths = nn.functional.embedding(env_ids, self.sketch_lengths)
        sketchs = sketchs[:, :sketch_lengths.max().item()]
        sketchs_emb = self.gru(self.sketch_embedding(sketchs))[0]
        batch_id = torch.arange(sketchs_emb.shape[0], device=sketchs_emb.device)
        final_states = sketchs_emb[batch_id, sketch_lengths - 1]
        return final_states


class EnvEmbedding(nn.Module):
    def __init__(self, emb_size):
        super(EnvEmbedding, self).__init__()
        self.embedding = nn.Embedding(len(env_list), emb_size)
        self.emb_size = emb_size

    def forward(self, env_ids):
        return self.embedding(env_ids)


class NoEnvEmbedding(nn.Module):
    def __init__(self, emb_size):
        super(NoEnvEmbedding, self).__init__()
        self.emb_size = emb_size
        self.zeros = nn.Parameter(torch.tensor(0.), requires_grad=False)

    def forward(self, env_ids):
        final_shape = list(env_ids.shape) + [self.emb_size]
        return self.zeros.repeat(*final_shape)


def get_env_encoder(env_arch, emb_size):
    if env_arch == 'sketch':
        return SketchEmbedding(emb_size)
    elif env_arch == 'grusketch':
        return GRUSketchEmbedding(emb_size)
    elif env_arch == 'emb':
        return EnvEmbedding(emb_size)
    elif env_arch == 'noenv':
        return NoEnvEmbedding(emb_size)
    else:
        raise ValueError
