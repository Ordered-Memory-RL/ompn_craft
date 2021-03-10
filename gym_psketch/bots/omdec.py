from ..utils import DictList
from .model_bot import ModelBot
from .utils import get_env_encoder
from ..env import Actions
import torch
import torch.nn as nn
from .om_utils import Distribution, Cell, ComCell, DecomCell

__all__ = ['OMDecBot', 'OMStackBot', 'OMStackNoRecurrBot']


class OMdecBase(ModelBot):
    """ Base class of OMdec """
    def __init__(self, nb_slots, slot_size, action_size, vec_size, env_arch, dropout=0.,
                 process='stickbreaking'):
        super(OMdecBase, self).__init__(vec_size=vec_size,
                                        hidden_size=slot_size)
        self.nb_slots = nb_slots
        self.slot_size = slot_size
        self.layernorm = nn.LayerNorm(slot_size, elementwise_affine=False)
        self.distribution = Distribution(slot_size * 4, slot_size, dropout=dropout, process=process)
        self.init_p = self.distribution.init_p(1, nslot=self.nb_slots)

        self.actor = nn.Sequential(nn.Linear(3 * slot_size, action_size - 1))
        self.critic = nn.Sequential(nn.Linear(3 * slot_size, 1))
        self.done_id = Actions.DONE.value
        self.memory_encoder = get_env_encoder(env_arch, slot_size)

    @property
    def is_recurrent(self):
        return True

    def step(self, obs, task, mems):
        """ Return output, mems, extra_info """
        raise NotImplementedError

    def forward(self, obs, env_ids, mems=None) -> DictList:
        task_emb = self.memory_encoder(env_ids)
        obs_inp = self.encode_obs(obs)
        output, memory, extra_info = self.step(obs_inp, task_emb, mems)
        output = torch.cat([output, task_emb, obs_inp], dim=-1)

        # Replace done with p_end
        p_action = torch.nn.functional.softmax(self.actor(output), dim=-1)
        p_end = extra_info['p_end'].clamp(1e-9, 1 - 1e-9)
        p_action = p_action * (1 - p_end)[:, None]
        p_action = torch.cat([p_action[:, :self.done_id], p_end[:, None], p_action[:, self.done_id:]], dim=1)
        values = self.critic(output)
        results = {'dist': torch.distributions.Categorical(probs=p_action), 'v': values, 'mems': memory}
        results.update(extra_info)
        return DictList(results)


class OMDecBot(OMdecBase):

    def __init__(self, vec_size, action_size, slot_size,
                 env_arch, nb_slots=3, dropout=0,
                 process='stickbreaking'):
        super(OMDecBot, self).__init__(vec_size=vec_size, action_size=action_size, slot_size=slot_size,
                                       nb_slots=nb_slots, dropout=dropout, process=process,
                                       env_arch=env_arch)
        self.cell = nn.ModuleList([Cell(hidden_size=slot_size, dropout=dropout, attn=False) for _ in range(nb_slots)])

    def step(self, input_enc, task_emb, memory):
        prev_m, prev_lc, prev_p = self._unflat_memory(memory)
        bsz, nslot, _ = prev_m.size()
        comb_input = torch.cat([input_enc, task_emb], dim=-1)

        p_hat = self.init_p.repeat([bsz, 1]).to(device=input_enc.device)
        not_use_init_id = (prev_p.sum(-1) != 0).nonzero().squeeze(1)
        if len(not_use_init_id) > 0:
            selected_inp = comb_input[not_use_init_id]
            selected_prev_m = prev_m[not_use_init_id]
            selected_prev_lc = prev_lc[not_use_init_id]
            dist_input = torch.cat([selected_inp[:, None, :].expand(-1, nslot, -1),
                                    selected_prev_m,
                                    selected_prev_lc], dim=-1)
            p_hat[not_use_init_id] = self.distribution(dist_input)
        p_end = p_hat[:, 0]
        p = nn.functional.normalize(p_hat[:, 1:], dim=1, p=1)
        cp = p.cumsum(dim=1)
        rcp = p.flip([1]).cumsum(dim=1).flip([1])

        lc_list = []
        rc_list = []
        lc = torch.zeros_like(prev_m[:, 0])
        for i in range(0, self.nb_slots):
            h = rcp[:, i, None] * prev_m[:, i] + (1 - rcp)[:, i, None] * lc
            lc, rc = self.cell[i](parent=h, inp_enc=comb_input)
            lc_list.append(lc)
            rc_list.append(rc)

        lc_array = torch.stack(lc_list, dim=1)
        rc_array = torch.stack(rc_list, dim=1)

        m = prev_m * (1 - cp)[:, :, None] + rc_array * cp[:, :, None]
        output = lc_array[:, -1]
        return output, self._flat_memory(m, lc_array, p), {'p': p_hat, 'p_end': p_end}

    def init_memory(self, env_ids):
        batch_size = env_ids.shape[0]
        device = env_ids.device
        first_slot = self.layernorm(self.memory_encoder(env_ids.long()))
        init_m = nn.functional.pad(first_slot[:, None, :], [0, 0, 0, self.nb_slots - 1], value=0.)
        init_lc = torch.zeros_like(init_m, device=device)
        init_p = torch.zeros([batch_size, self.nb_slots], device=device)
        return torch.cat([init_m.reshape(batch_size, -1),
                          init_lc.reshape(batch_size, -1),
                          init_p], dim=-1)

    def _flat_memory(self, mem, lc, p):
        batch_size = mem.shape[0]
        mem_size = self.nb_slots * self.slot_size
        return torch.cat([mem.reshape(batch_size, mem_size),
                          lc.reshape(batch_size, mem_size),
                          p.reshape(batch_size, self.nb_slots)], dim=1)

    def _unflat_memory(self, memory):
        mem_size = self.nb_slots * self.slot_size
        mem = memory[:, :mem_size].reshape(-1, self.nb_slots, self.slot_size)
        lc = memory[:, mem_size: 2 * mem_size].reshape(-1, self.nb_slots,
                                                       self.slot_size)
        p = memory[:, 2 * mem_size:]
        return mem, lc, p


class OMStackBot(OMdecBase):
    def __init__(self, vec_size, action_size, slot_size, env_arch, nb_slots=3, dropout=0,
                 process='stickbreaking'):
        super(OMStackBot, self).__init__(vec_size=vec_size, action_size=action_size, slot_size=slot_size,
                                         nb_slots=nb_slots, dropout=dropout, process=process,
                                         env_arch=env_arch)
        self.com_cell = nn.ModuleList(
            [ComCell(hidden_size=slot_size, dropout=dropout) for _ in range(nb_slots)])
        self.decom_cell = nn.ModuleList(
            [DecomCell(hidden_size=slot_size, dropout=dropout) for _ in range(nb_slots)])

    def step(self, input_enc, task_emb, memory):
        prev_m, prev_p = self._unflat_memory(memory)
        bsz, nslot, _ = prev_m.size()
        comb_input = torch.cat([input_enc, task_emb], dim=-1)

        p_hat = self.init_p.repeat([bsz, 1]).to(device=input_enc.device)
        cand_m = prev_m
        not_init_id = (prev_p.sum(-1) != 0).nonzero().squeeze(1)
        if len(not_init_id) > 0:
            cm_list = []
            selected_inp = comb_input[not_init_id]
            selected_prev_m = prev_m[not_init_id]
            h = input_enc[not_init_id]
            for i in range(self.nb_slots - 1, -1, -1):
                h = self.com_cell[i](h, selected_prev_m[:, i, :], selected_inp)
                cm_list.append(h)
            selected_cand_m = torch.stack(cm_list[::-1], dim=1)
            cand_m[not_init_id] = selected_cand_m

            dist_input = torch.cat([selected_inp[:, None, :].expand(-1, nslot, -1),
                                    selected_prev_m,
                                    selected_cand_m], dim=-1)
            p_hat[not_init_id] = self.distribution(dist_input)

        p_end = p_hat[:, 0]
        p = torch.nn.functional.normalize(p_hat[:, 1:], dim=1, p=1)
        cp = p.cumsum(dim=1)
        rcp = p.flip([1]).cumsum(dim=1).flip([1])

        chl = torch.zeros_like(cand_m[:, 0])
        chl_list = []
        for i in range(self.nb_slots):
            chl_list.append(chl)
            h = rcp[:, i, None] * cand_m[:, i] + (1 - rcp)[:, i, None] * chl
            chl = self.decom_cell[i](parent=h, inp_enc=comb_input, context=None)
        chl_array = torch.stack(chl_list, dim=1)

        m = prev_m * (1 - cp)[:, :, None] + cand_m * p[:, :, None] + chl_array * (1 - rcp)[:, :, None]
        output = m[:, -1]
        return output, self._flat_memory(m, p), {'p': p_hat, 'p_end': p_end}

    def _flat_memory(self, mem, p):
        batch_size = mem.shape[0]
        mem_size = self.nb_slots * self.slot_size
        return torch.cat([mem.reshape(batch_size, mem_size),
                          p.reshape(batch_size, self.nb_slots)], dim=1)

    def _unflat_memory(self, memory):
        mem_size = self.nb_slots * self.slot_size
        mem = memory[:, :mem_size].reshape(-1, self.nb_slots, self.slot_size)
        p = memory[:, mem_size:]
        return mem, p

    def init_memory(self, env_ids):
        batch_size = env_ids.shape[0]
        device = env_ids.device
        first_slot = self.layernorm(self.memory_encoder(env_ids.long()))
        init_m = nn.functional.pad(first_slot[:, None, :], [0, 0, 0, self.nb_slots - 1], value=0.)
        #init_m = torch.zeros([batch_size, self.slot_size * self.nb_slots], device=device)
        init_p = torch.zeros([batch_size, self.nb_slots], device=device)
        return torch.cat([init_m.reshape(batch_size, -1),
                          init_p], dim=-1)


class OMStackNoRecurrBot(OMdecBase):
    def __init__(self, vec_size, action_size, slot_size, env_arch, nb_slots=3, dropout=0,
                 process='stickbreaking'):
        super(OMStackNoRecurrBot, self).__init__(vec_size=vec_size, action_size=action_size, slot_size=slot_size,
                                         nb_slots=nb_slots, dropout=dropout, process=process,
                                         env_arch=env_arch)
        self.com_cell = nn.ModuleList(
            [ComCell(hidden_size=slot_size, dropout=dropout) for _ in range(nb_slots)])
        self.decom_cell = nn.ModuleList(
            [DecomCell(hidden_size=slot_size, dropout=dropout) for _ in range(nb_slots)])

    def step(self, input_enc, task_emb, memory):
        prev_m, prev_p = self._unflat_memory(memory)
        bsz, nslot, _ = prev_m.size()
        comb_input = torch.cat([input_enc, task_emb], dim=-1)

        p_hat = self.init_p.repeat([bsz, 1]).to(device=input_enc.device)
        cand_m = prev_m
        not_init_id = (prev_p.sum(-1) != 0).nonzero().squeeze(1)
        if len(not_init_id) > 0:
            cm_list = []
            selected_inp = comb_input[not_init_id]
            selected_prev_m = prev_m[not_init_id]
            for i in range(self.nb_slots - 1, -1, -1):
                h = self.com_cell[i](torch.zeros_like(input_enc[not_init_id]),
                                     selected_prev_m[:, i, :],
                                     selected_inp)
                cm_list.append(h)
            selected_cand_m = torch.stack(cm_list[::-1], dim=1)
            cand_m[not_init_id] = selected_cand_m

            dist_input = torch.cat([selected_inp[:, None, :].expand(-1, nslot, -1),
                                    selected_prev_m,
                                    selected_cand_m], dim=-1)
            p_hat[not_init_id] = self.distribution(dist_input)

        p_end = p_hat[:, 0]
        p = torch.nn.functional.normalize(p_hat[:, 1:], dim=1, p=1)
        cp = p.cumsum(dim=1)
        rcp = p.flip([1]).cumsum(dim=1).flip([1])

        chl = torch.zeros_like(cand_m[:, 0])
        chl_list = []
        for i in range(self.nb_slots):
            chl_list.append(chl)
            h = rcp[:, i, None] * cand_m[:, i] + (1 - rcp)[:, i, None] * chl
            chl = self.decom_cell[i](parent=h, inp_enc=comb_input, context=None)
        chl_array = torch.stack(chl_list, dim=1)

        m = prev_m * (1 - cp)[:, :, None] + cand_m * p[:, :, None] + chl_array * (1 - rcp)[:, :, None]
        output = m[:, -1]
        return output, self._flat_memory(m, p), {'p': p_hat, 'p_end': p_end}

    def _flat_memory(self, mem, p):
        batch_size = mem.shape[0]
        mem_size = self.nb_slots * self.slot_size
        return torch.cat([mem.reshape(batch_size, mem_size),
                          p.reshape(batch_size, self.nb_slots)], dim=1)

    def _unflat_memory(self, memory):
        mem_size = self.nb_slots * self.slot_size
        mem = memory[:, :mem_size].reshape(-1, self.nb_slots, self.slot_size)
        p = memory[:, mem_size:]
        return mem, p

    def init_memory(self, env_ids):
        batch_size = env_ids.shape[0]
        device = env_ids.device
        first_slot = self.layernorm(self.memory_encoder(env_ids.long()))
        init_m = nn.functional.pad(first_slot[:, None, :], [0, 0, 0, self.nb_slots - 1], value=0.)
        #init_m = torch.zeros([batch_size, self.slot_size * self.nb_slots], device=device)
        init_p = torch.zeros([batch_size, self.nb_slots], device=device)
        return torch.cat([init_m.reshape(batch_size, -1),
                          init_p], dim=-1)
