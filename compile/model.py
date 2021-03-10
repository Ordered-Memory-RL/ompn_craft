"""
Adapted from https://github.com/tkipf/compile/blob/master/modules.py
"""
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from gym_psketch.bots.utils import get_env_encoder
from gym_psketch import DictList

EPS = 1e-17
NEG_INF = -1e30


def to_one_hot(indices, max_index):
    """Get one-hot encoding of index tensors."""
    zeros = torch.zeros(
        indices.size()[0], max_index, dtype=torch.float32,
        device=indices.device)
    return zeros.scatter_(1, indices.unsqueeze(1), 1)


def gumbel_sample(shape):
    """Sample Gumbel noise."""
    uniform = torch.rand(shape).float()
    return - torch.log(EPS - torch.log(uniform + EPS))


def gumbel_softmax_sample(logits, temp=1.):
    """Sample from the Gumbel softmax / concrete distribution."""
    gumbel_noise = gumbel_sample(logits.size())
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    return F.softmax((logits + gumbel_noise) / temp, dim=-1)


def gaussian_sample(mu, log_var):
    """Sample from Gaussian distribution."""
    gaussian_noise = torch.randn(mu.size())
    if mu.is_cuda:
        gaussian_noise = gaussian_noise.cuda()
    return mu + torch.exp(log_var * 0.5) * gaussian_noise


def kl_gaussian(mu, log_var):
    """KL divergence between Gaussian posterior and standard normal prior."""
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)


def kl_categorical_uniform(preds):
    """KL divergence between categorical distribution and uniform prior."""
    kl_div = preds * torch.log(preds + EPS)  # Constant term omitted.
    return kl_div.sum(1)


def kl_categorical(preds, log_prior):
    """KL divergence between two categorical distributions."""
    kl_div = preds * (torch.log(preds + EPS) - log_prior)
    return kl_div.sum(1)


def poisson_categorical_log_prior(length, rate, device):
    """Categorical prior populated with log probabilities of Poisson dist."""
    rate = torch.tensor(rate, dtype=torch.float32, device=device)
    values = torch.arange(
        1, length + 1, dtype=torch.float32, device=device).unsqueeze(0)
    log_prob_unnormalized = torch.log(
        rate) * values - rate - (values + 1).lgamma()
    # TODO(tkipf): Length-sensitive normalization.
    return F.log_softmax(log_prob_unnormalized, dim=1)  # Normalize.


def log_cumsum(probs, dim=1):
    """Calculate log of inclusive cumsum."""
    return torch.log(torch.cumsum(probs, dim=dim) + EPS)


def get_lstm_initial_state(batch_size, hidden_dim, device):
    """Get empty (zero) initial states for LSTM."""
    hidden_state = torch.zeros(batch_size, hidden_dim, device=device)
    cell_state = torch.zeros(batch_size, hidden_dim, device=device)
    return hidden_state, cell_state


def get_segment_probs(all_b_samples, all_masks, segment_id):
    """Get segment probabilities for a particular segment ID."""
    neg_cumsum = 1 - torch.cumsum(all_b_samples[segment_id], dim=1)
    if segment_id > 0:
        return neg_cumsum * all_masks[segment_id - 1]
    else:
        return neg_cumsum


class CompILE(nn.Module):
    """CompILE example implementation."""
    def __init__(self, vec_size, hidden_size, action_size,
                 env_arch, max_num_segments, latent_dist='discrete',
                 temp_b=1., temp_z=1., beta_b=0.1, beta_z=0.1, prior_rate=3.):
        super(CompILE, self).__init__()
        self.input_dim = vec_size
        self.hidden_dim = hidden_size
        self.latent_dim = hidden_size
        self.max_num_segments = max_num_segments
        self.temp_b = temp_b
        self.temp_z = temp_z
        self.action_size = action_size
        self.latent_dist = latent_dist

        # For loss
        self.beta_b = beta_b
        self.beta_z = beta_z
        self.prior_rate = prior_rate

        # encoder
        self.obs_enc = nn.Linear(vec_size, hidden_size)
        self.action_emb = nn.Embedding(action_size, hidden_size)
        self.env_emb = get_env_encoder(env_arch, hidden_size)
        self.input_encoder = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size, elementwise_affine=False)
        )


        self.lstm_cell = nn.LSTMCell(hidden_size, hidden_size)

        # LSTM output heads.
        self.head_z_1 = nn.Linear(hidden_size, hidden_size)  # Latents (z).

        if latent_dist == 'gaussian':
            self.head_z_2 = nn.Linear(hidden_size, hidden_size * 2)
        elif latent_dist == 'concrete':
            self.head_z_2 = nn.Linear(hidden_size, hidden_size)
        else:
            raise ValueError('Invalid argument for `latent_dist`.')

        self.head_b_1 = nn.Linear(hidden_size, hidden_size)  # Boundaries (b).
        self.head_b_2 = nn.Linear(hidden_size, 1)

        # Decoder p(a | s, env_id, z)
        self.decoder = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def get_reconstruction_accuracy(self, batch, all_b, all_recs):
        """Calculate reconstruction accuracy (averaged over sequence length)."""
        inputs = batch.action
        batch_size = inputs.size(0)

        rec_seq = []
        rec_acc = 0.
        for sample_idx in range(batch_size):
            prev_boundary_pos = 0
            rec_seq_parts = []
            for seg_id in range(self.max_num_segments):
                boundary_pos = torch.argmax(
                    all_b['samples'][seg_id], dim=-1)[sample_idx]
                if prev_boundary_pos > boundary_pos:
                    boundary_pos = prev_boundary_pos
                seg_rec_seq = torch.argmax(all_recs[seg_id], dim=-1)
                rec_seq_parts.append(
                    seg_rec_seq[sample_idx, prev_boundary_pos:boundary_pos])
                prev_boundary_pos = boundary_pos
            rec_seq.append(torch.cat(rec_seq_parts))
            cur_length = rec_seq[sample_idx].size(0)
            matches = rec_seq[sample_idx] == inputs[sample_idx, :cur_length]
            rec_acc += matches.float().mean()
        rec_acc /= batch_size
        return {'reconst_acc': rec_acc}

    def compute_loss(self, batch, all_b, all_masks, all_recs, all_z):
        # Compute loss
        inp = batch.action
        targets = inp.view(-1)
        nll = 0.
        kl_z = 0.
        for seg_id in range(self.max_num_segments):
            seg_prob = get_segment_probs(
                all_b['samples'], all_masks, seg_id)
            logits = all_recs[seg_id].view(-1, self.action_size)
            seg_loss = F.cross_entropy(
                logits, targets, reduction='none').view(-1, inp.size(1))

            # Ignore EOS token (last sequence element) in loss.
            nll += (seg_loss[:, :-1] * seg_prob[:, :-1]).sum(1).mean(0)

            # KL divergence on z.
            if self.latent_dist == 'gaussian':
                mu, log_var = torch.split(
                    all_z['logits'][seg_id], self.latent_dim, dim=1)
                kl_z += kl_gaussian(mu, log_var).mean(0)
            elif self.latent_dist == 'concrete':
                kl_z += kl_categorical_uniform(
                    F.softmax(all_z['logits'][seg_id], dim=-1)).mean(0)
            else:
                raise ValueError('Invalid argument for `latent_dist`.')

        # KL divergence on b (first segment only, ignore first time step).
        # TODO(tkipf): Implement alternative prior on soft segment length.
        probs_b = F.softmax(all_b['logits'][0], dim=-1)
        log_prior_b = poisson_categorical_log_prior(
            probs_b.size(1), self.prior_rate, device=inp.device)
        kl_b = self.max_num_segments * kl_categorical(
            probs_b[:, 1:], log_prior_b[:, 1:]).mean(0)

        loss = nll + self.beta_z * kl_z + self.beta_b * kl_b
        return {'loss': loss, 'nll': nll, 'kl_b': kl_b, 'kl_z': kl_z}

    def forward(self, batch, lengths):
        # Encoding inputs
        embeddings = self.encode_obs(batch)

        # Create initial mask.
        mask = torch.ones(embeddings.size(0), embeddings.size(1), device=embeddings.device)

        all_b = {'logits': [], 'samples': []}
        all_z = {'logits': [], 'samples': []}
        all_encs = []
        all_recs = []
        all_masks = []
        for seg_id in range(self.max_num_segments):

            # Get masked LSTM encodings of inputs.
            encodings = self.masked_encode(embeddings, mask)
            all_encs.append(encodings)

            # Get boundaries (b) for current segment.
            logits_b, sample_b = self.get_boundaries(
                encodings, seg_id, lengths)
            all_b['logits'].append(logits_b)
            all_b['samples'].append(sample_b)

            # Get latents (z) for current segment.
            logits_z, sample_z = self.get_latents(
                encodings, sample_b)
            all_z['logits'].append(logits_z)
            all_z['samples'].append(sample_z)

            # Get masks for next segment.
            mask = self.get_next_masks(all_b['samples'])
            all_masks.append(mask)

            # Decode current segment from latents (z).
            reconstructions = self.decode(batch, sample_z)
            all_recs.append(reconstructions)

        # loss
        result = self.compute_loss(batch, all_b, all_masks, all_recs, all_z)
        result.update(self.get_reconstruction_accuracy(batch, all_b, all_recs))

        # accuracy
        return result, {'segment': all_b['samples']}

    def encode_obs(self, obs: DictList):
        obs_emb = self.obs_enc(obs.features.float())
        env_emb = self.env_emb(obs.env_id)
        action_emb = self.action_emb(obs.action)
        combined_input = torch.cat([obs_emb, env_emb, action_emb], dim=-1)
        return self.input_encoder(combined_input)

    def masked_encode(self, inputs, mask):
        """Run masked RNN encoder on input sequence."""
        hidden = get_lstm_initial_state(
            inputs.size(0), self.hidden_dim, device=inputs.device)
        outputs = []
        for step in range(inputs.size(1)):
            hidden = self.lstm_cell(inputs[:, step], hidden)
            hidden = (mask[:, step, None] * hidden[0],
                      mask[:, step, None] * hidden[1])  # Apply mask.
            outputs.append(hidden[0])
        return torch.stack(outputs, dim=1)

    def get_boundaries(self, encodings, segment_id, lengths):
        """Get boundaries (b) for a single segment in batch."""
        if segment_id == self.max_num_segments - 1:
            # Last boundary is always placed on last sequence element.
            logits_b = None
            sample_b = torch.zeros_like(encodings[:, :, 0]).scatter_(
                1, lengths.unsqueeze(1) - 1, 1)
        else:
            hidden = F.relu(self.head_b_1(encodings))
            logits_b = self.head_b_2(hidden).squeeze(-1)
            # Mask out first position with large neg. value.
            neg_inf = torch.ones(
                encodings.size(0), 1, device=encodings.device) * NEG_INF
            # TODO(tkipf): Mask out padded positions with large neg. value.
            length_mask = torch.arange(lengths.max().item(), device=lengths.device)[None, :] >= lengths[:, None]
            logits_b = logits_b.masked_fill(length_mask, NEG_INF)
            logits_b = torch.cat([neg_inf, logits_b[:, 1:]], dim=1)
            if self.training:
                sample_b = gumbel_softmax_sample(
                    logits_b, temp=self.temp_b)
            else:
                sample_b_idx = torch.argmax(logits_b, dim=1)
                sample_b = to_one_hot(sample_b_idx, logits_b.size(1))

        return logits_b, sample_b

    def get_latents(self, encodings, probs_b):
        """Read out latents (z) form input encodings for a single segment."""
        readout_mask = probs_b[:, 1:, None]  # Offset readout by 1 to left.
        readout = (encodings[:, :-1] * readout_mask).sum(1)
        hidden = F.relu(self.head_z_1(readout))
        logits_z = self.head_z_2(hidden)

        # Gaussian latents.
        if self.latent_dist == 'gaussian':
            if self.training:
                mu, log_var = torch.split(logits_z, self.latent_dim, dim=1)
                sample_z = gaussian_sample(mu, log_var)
            else:
                sample_z = logits_z[:, :self.latent_dim]

        # Concrete / Gumbel softmax latents.
        elif self.latent_dist == 'concrete':
            if self.training:
                sample_z = gumbel_softmax_sample(
                    logits_z, temp=self.temp_z)
            else:
                sample_z_idx = torch.argmax(logits_z, dim=1)
                sample_z = to_one_hot(sample_z_idx, logits_z.size(1))
        else:
            raise ValueError('Invalid argument for `latent_dist`.')

        return logits_z, sample_z

    def decode(self, obs: DictList, sample_z):
        """Decode single time step from latents and repeat over full seq."""
        obs_emb = self.obs_enc(obs.features.float())
        env_emb = self.env_emb(obs.env_id)
        comb_input = torch.cat([obs_emb, env_emb,
                                sample_z[:, None, :].repeat(1, env_emb.shape[1], 1)], dim=-1)
        return self.decoder(comb_input)

    def get_next_masks(self, all_b_samples):
        """Get RNN hidden state masks for next segment."""
        if len(all_b_samples) < self.max_num_segments:
            # Product over cumsums (via log->sum->exp).
            log_cumsums = list(
                map(lambda x: log_cumsum(x, dim=1), all_b_samples))
            mask = torch.exp(sum(log_cumsums))
            return mask
        else:
            return None
