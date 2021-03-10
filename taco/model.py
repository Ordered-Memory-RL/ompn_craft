"""
Modular policy
https://github.com/KyriacosShiarli/taco
"""
from torch import nn
import torch
from gym_psketch import DictList, ID2SKETCHS


class HMLP(nn.Module):
    def __init__(self, in_dim, n_actions, num_units=[400, 100]):
        super(HMLP, self).__init__()
        actor_layers = []
        stop_layers = []
        for out_dim in num_units:
            actor_layers.append(nn.Linear(in_dim, out_dim))
            stop_layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim

        # Final layers
        actor_layers.append(nn.Linear(in_dim, n_actions))
        stop_layers.append(nn.Linear(in_dim, 2))
        self.actor_layers = nn.ModuleList(actor_layers)
        self.stop_layers = nn.ModuleList(stop_layers)
        self.depth = len(num_units)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs, dropout_rate=0.):
        actor_output = inputs
        stop_output = inputs
        for i in range(self.depth):
            actor_output = self.relu(self.actor_layers[i](actor_output))
            stop_output = self.relu(self.stop_layers[i](stop_output))
            stop_output = nn.functional.dropout(stop_output, p=dropout_rate, training=self.training)
        action_logits = self.actor_layers[-1](actor_output)
        stop_logits = self.stop_layers[-1](stop_output)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        stop_dist = torch.distributions.Categorical(logits=stop_logits)
        return DictList({'action_dist': action_dist, 'stop_dist': stop_dist})

    def get_action(self, inputs, mode='greedy'):
        dists = self.forward(inputs)
        if mode == 'greedy':
            action = dists.action_dist.logits.argmax(-1)
            stop = dists.stop_dist.logits.argmax(-1)
        elif mode == 'sample':
            action = dists.action_dist.sample()
            stop = dists.stop_dist.sample()
        else:
            raise ValueError
        return DictList({'action': action, 'stop': stop})


class ModularPolicy(nn.Module):
    def __init__(self, nb_subtasks, input_dim, n_actions):
        super(ModularPolicy, self).__init__()
        self.mlps = nn.ModuleList([HMLP(in_dim=input_dim, n_actions=n_actions) for _ in range(nb_subtasks)])

        self.candidates = None
        self.sketch_id = None

    def env_id_to_mlps(self, env_id):
        subtasks = ID2SKETCHS[env_id]
        return [self.mlps[subtask] for subtask in subtasks]

    def reset(self, env_id):
        self.sketch_id = 0
        self.candidates = self.env_id_to_mlps(env_id)

    def get_action(self, inputs, mode='greedy'):
        action = None
        while True:
            if self.sketch_id == len(self.candidates):
                break

            curr_mlp = self.candidates[self.sketch_id]
            action_outputs = curr_mlp.get_action(inputs, mode)
            stop = action_outputs.stop.item()
            if stop == 1:
                self.sketch_id += 1
            else:
                action = action_outputs.action.item()
                break

        return action

    def forward(self, task, states, actions, dropout_p=0.):
        """
        Args:
            task:  int
            states:  [bsz, in_dim]
            actions: [bsz]

        Returns:
            action_logprobs [bsz]
            stop_logprobs [bsz, 2]
        """
        mlp = self.mlps[task]
        model_output = mlp(states, dropout_p)
        action_logprobs = model_output.action_dist.log_prob(actions)
        stop_logprobs = torch.nn.functional.log_softmax(model_output.stop_dist.logits, dim=-1)
        return {'action_logprobs': action_logprobs, 'stop_logprobs': stop_logprobs}

