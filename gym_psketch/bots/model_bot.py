import torch
import torch.nn as nn

from gym_psketch.bots.utils import get_env_encoder

__all__ = ['MLPBot', 'LSTMBot', 'ModelBot']

from gym_psketch.utils import DictList


class ModelBot(torch.nn.Module):
    def __init__(self, vec_size, hidden_size):
        super(ModelBot, self).__init__()
        self.vec_size = vec_size
        self.obs_size = hidden_size

        # Image encoder
        self.inp_enc = nn.Linear(self.vec_size, hidden_size)

    def forward(self, obs: DictList, env_ids, mems=None) -> DictList:
        """ Single step forward (Imitate babyAI)
        :param obs:
        :param env_ids: env ids
        :param mems: [bsz, mem_size] or None
        :return DictList with keys "dist", "v", "mems" (optional)
        """
        raise NotImplementedError

    def encode_obs(self, obs: DictList):
        return self.inp_enc(obs.features.float())

    def get_action(self, obs: DictList, env_ids, mems, mode='greedy') -> DictList:
        """
        obs: [bsz, obs_size]
        mode: "greedy" | "sample"
        """
        output = self.forward(obs, env_ids, mems)
        dist = output.dist
        if mode == 'greedy':
            output.actions = torch.argmax(dist.logits, -1)
        elif mode == 'sample':
            output.actions = dist.sample()
        else:
            raise ValueError
        return output

    @property
    def is_recurrent(self):
        return False

    def init_memory(self, env_ids):
        """ Return initial memory """
        return None


class MLPBot(ModelBot):
    def __init__(self, vec_size, action_size, hidden_size, env_arch):
        super(MLPBot, self).__init__(vec_size=vec_size, hidden_size=hidden_size)
        self.actor = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size),
                                   nn.Tanh(),
                                   nn.Linear(hidden_size, action_size))
        self.critic = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size),
                                    nn.Tanh(),
                                    nn.Linear(hidden_size, 1))
        self.env_emb = get_env_encoder(env_arch, hidden_size)

    def forward(self, obs, env_ids, mems=None) -> DictList:
        obs_repr = self.encode_obs(obs)
        env_emb = self.env_emb(env_ids)
        inp = torch.cat([obs_repr, env_emb], dim=-1)
        logits = self.actor(inp)
        values = self.critic(inp)
        return DictList({'v': values, 'dist': torch.distributions.Categorical(logits=logits)})


class LSTMBot(ModelBot):
    def __init__(self, vec_size, action_size, hidden_size, env_arch, num_layers):
        super(LSTMBot, self).__init__(vec_size=vec_size,
                                      hidden_size=hidden_size)
        self.env_emb = get_env_encoder(env_arch, hidden_size)
        self.actor = nn.Sequential(nn.Linear(3 * hidden_size, action_size))
        self.critic = nn.Sequential(nn.Linear(3 * hidden_size, 1))
        self.lstm = torch.nn.LSTM(input_size=hidden_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  batch_first=True)
        self.layernorm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        mem_size = 2*num_layers*hidden_size
        self.memory_encoder = get_env_encoder(env_arch, mem_size)

    @property
    def is_recurrent(self):
        return True

    def _flat_mem(self, lstm_mems):
        # [num_layers, bsz, hidden_size, 2]
        mems = torch.stack(lstm_mems, -1)

        # [bsz, num_layers, hidden_size, 2]
        mems = mems.permute(1, 0, 2, 3)
        return mems.reshape(mems.shape[0], -1)

    def _unflat_mem(self, mems):
        """
        :param mems: [bsz, mem_size]
        :return:
        """
        # bsz, num_layers, hidden_size, 2
        mems = mems.view(mems.shape[0], self.lstm.num_layers, self.lstm.hidden_size, 2)
        mems = mems.permute(1, 0, 2, 3)
        lstm_mems = mems.chunk(2, dim=-1)
        return lstm_mems[0].squeeze(-1).contiguous(), lstm_mems[1].squeeze(-1).contiguous()

    def init_memory(self, env_ids):
        """
        :param env_ids: [bsz]
        :return: init_mem: [bsz, mem_size]
        """
        return self.memory_encoder(env_ids)

    def forward(self, obs, env_ids, mems=None) -> DictList:
        """
        :param obss: [bsz, obs_size]
        :param mems: [bsz, mem_size]
        :return:
        """
        inputs = self.layernorm(self.encode_obs(obs))
        lstm_mems = self._unflat_mem(mems)
        outputs, next_lstm_mems = self.lstm(inputs.unsqueeze(1), lstm_mems)
        next_mems = self._flat_mem(next_lstm_mems)
        outputs = outputs.squeeze(1)
        outputs = torch.cat([outputs, self.env_emb(env_ids), inputs], dim=-1)
        logits = self.actor(outputs)
        values = self.critic(outputs)
        results = {'mems': next_mems,
                   'dist': torch.distributions.Categorical(logits=logits),
                   'v': values}
        return DictList(results)
