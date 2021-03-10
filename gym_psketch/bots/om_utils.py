import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Distribution(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, process='softmax'):
        super(Distribution, self).__init__()

        assert process in ['stickbreaking', 'softmax']

        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

        self.hidden_size = hidden_size
        self.process_name = process

    def init_p(self, bsz, nslot):
        weight = next(self.parameters()).data
        p = weight.new(bsz, nslot + 1).zero_()
        p[:, 1] = 1
        return p

    @staticmethod
    def process_stickbreaking(beta):
        beta = beta.flip([1])
        y = (1 - beta).cumprod(-1)
        p = F.pad(beta, (0, 1), value=1) * F.pad(y, (1, 0), value=1)
        p = p.flip([1])
        return p

    @staticmethod
    def process_softmax(beta, mask):
        nslot = beta.size(1)
        beta = F.pad(beta, (1, 0), value=0)
        beta_normalized = beta - beta.max(dim=-1)[0][:, None]
        x = torch.exp(beta_normalized)
        if mask is not None:
            x = x[:, None, :].expand(-1, nslot, -1)
            x = x.triu(diagonal=1)
            p_candidates = F.normalize(x, p=1, dim=2)
            p = torch.bmm(mask[:, None, :], p_candidates).squeeze(dim=1)
        else:
            p = F.normalize(x, p=1, dim=1)
        return p

    def forward(self, input, mask=None):
        beta = self.mlp(input).squeeze(dim=2)
        if self.process_name == 'stickbreaking':
            beta = torch.sigmoid(beta)
            # if mask is not None:
            #     beta = beta * mask.cumsum(dim=-1)
            return self.process_stickbreaking(beta)
        elif self.process_name == 'softmax':
            beta = beta / math.sqrt(self.hidden_size)
            return self.process_softmax(beta, mask)


class Attention(nn.Module):
    def __init__(self, hidden_size, dropout=0.2):
        super(Attention, self).__init__()
        self.value = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.LayerNorm(hidden_size)
        self.gating = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        self.hidden_size = hidden_size
        self.drop = nn.Dropout(dropout)

    def forward(self, input, encoded):
        memory, memory_mask = encoded
        batch_size, _, _ = input.size()
        key_count, batch_size, _ = memory.size()
        query = self.query(input)
        key = self.key(memory)
        value = self.value(memory)
        scores = torch.einsum('bnd,btd->bnt', (query, key)) / (self.hidden_size ** 0.5)

        scores.masked_fill_(~(memory_mask[:, None, :].bool()), -float('inf'))
        attn = torch.softmax(scores, dim=-1)
        context = torch.einsum('bnt,btd->bnd', (attn, value))

        g_context = self.gating(torch.cat([input, context], dim=-1))
        output = input + g_context * context

        return output


class Cell(nn.Module):
    def __init__(self, hidden_size, dropout, attn=False):
        super(Cell, self).__init__()
        self.hidden_size = hidden_size
        self.cell_hidden_size = 4 * hidden_size

        if attn:
            self.attn = Attention(hidden_size=hidden_size)
        else:
            self.attn = None

        self.input_t = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(3 * hidden_size, self.cell_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.cell_hidden_size, 6 * hidden_size),
        )

        self.gates = nn.Sequential(
            nn.LayerNorm(hidden_size * 4),
            nn.Sigmoid()
        )

        self.drop = nn.Dropout(dropout)

        self.activation = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(self, inp_enc, parent, context=None):
        g_input, c_input = self.input_t(torch.cat([parent, inp_enc], dim=1)).split(
            (self.hidden_size * 4,
             self.hidden_size * 2),
            dim=-1
        )

        lgate, rgate, lcgate, rcgate = self.gates(g_input).chunk(4, dim=-1)
        lcell, rcell = c_input.chunk(2, dim=-1)

        lchild = self.activation(lgate * self.drop(parent) + lcgate * lcell)
        rchild = self.activation(rgate * self.drop(parent) + rcgate * rcell)

        if self.attn is not None and context is not None:
            children = torch.stack([lchild, rchild], dim=1)
            children = self.attn(children, context).squeeze(dim=1)
            lchild, rchild = torch.unbind(children, dim=1)

        return lchild, rchild


class ComCell(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(ComCell, self).__init__()
        self.hidden_size = hidden_size
        self.cell_hidden_size = 4 * hidden_size

        self.input_t = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, self.cell_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.cell_hidden_size, hidden_size * 4),
        )

        self.gates = nn.Sequential(
            nn.LayerNorm(hidden_size * 3),
            nn.Sigmoid(),
        )

        self.activation = nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.drop = nn.Dropout(dropout)

    def forward(self, vi, hi, obs):
        input = torch.cat([vi, hi, obs], dim=-1)

        g_input, cell = self.input_t(input).split(
            (self.hidden_size * 3, self.hidden_size),
            dim=-1
        )

        gates = self.gates(g_input)
        vg, hg, cg = gates.chunk(3, dim=1)
        output = self.activation(vg * vi + hg * hi + cg * cell)
        return output


class DecomCell(nn.Module):
    def __init__(self, hidden_size, dropout, attn=False):
        super(DecomCell, self).__init__()
        self.hidden_size = hidden_size
        self.cell_hidden_size = 4 * hidden_size

        if attn:
            self.attn = Attention(hidden_size=hidden_size)
        else:
            self.attn = None

        self.input_t = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(3 * hidden_size, self.cell_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.cell_hidden_size, 3 * hidden_size),
        )

        self.gates = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Sigmoid()
        )

        self.drop = nn.Dropout(dropout)

        self.activation = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(self, inp_enc, parent, context=None):
        g_input, cell = self.input_t(torch.cat([parent, inp_enc], dim=1)).split(
            (self.hidden_size * 2,
             self.hidden_size * 1),
            dim=-1
        )

        gate, cgate = self.gates(g_input).chunk(2, dim=-1)

        child = self.activation(gate * self.drop(parent) + cgate * cell)

        if self.attn is not None and context is not None:
            child = self.attn(child, context).squeeze(dim=1)

        return child
