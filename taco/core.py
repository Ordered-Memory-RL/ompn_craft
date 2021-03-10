import numpy as np
import torch

from gym_psketch import DictList, CraftWorld, Actions, ID2SKETCHS
from gym_psketch.evaluate import get_subtask_seq
from taco.model import ModularPolicy

NEG_INF = -100000000.
ZERO = 1e-32


def tac_forward(action_logprobs, stop_logprobs, lengths, subtask_lengths):
    """
    Args:
        action_logprobs: [bsz, len, tasks]
        stop_logprobs: [bsz, len, tasks,  2]
        lengths: [bsz]
        subtask_lengths [bsz]
    Returns:
        log_alphas [bsz, len, tasks]
        dirs [bsz, len, tasks]: If dirs[bsz, t, task] = 0, then alpha[b, t, task] is formed by nonstop.
        otherwise it's the descendent of child.
    """
    bsz, length, nb_tasks = action_logprobs.shape[0], action_logprobs.shape[1], action_logprobs.shape[2]

    # Initial log alpha
    init_alpha = torch.zeros([bsz, nb_tasks], device=action_logprobs.device)
    init_alpha[:, 0] = action_logprobs[:, 0, 0].exp()
    zeros = torch.tensor(ZERO, device=init_alpha.device)
    alphas = [init_alpha]
    init_dir = torch.zeros([bsz, nb_tasks], device=action_logprobs.device)
    dirs = [init_dir]
    action_probs = action_logprobs.exp()
    stop_probs = stop_logprobs.exp()
    for t in range(1, length):
        prev_alpha = alphas[-1] # [bsz, nb_task]
        nonstop = prev_alpha * stop_probs[:, t, :, 0]
        stop = prev_alpha * stop_probs[:, t, :, 1]
        stop = torch.cat([zeros.repeat(bsz, 1), stop[:, :-1]], dim=-1)
        untillnow = nonstop + stop
        curr_dir = (stop > nonstop).float()
        alpha = action_probs[:, t, :] * untillnow
        subtask_mask = torch.arange(subtask_lengths.max(),
                                    device=subtask_lengths.device)[None, :] < subtask_lengths[:, None]
        alpha = alpha.masked_fill(~subtask_mask, zeros)

        # prevent underflow
        alpha = alpha / torch.sum(alpha, dim=-1, keepdim=True)

        # Mask time
        alpha = torch.where((t < lengths).unsqueeze(-1), alpha, zeros.repeat(bsz, nb_tasks))
        curr_dir = torch.where((t < lengths).unsqueeze(-1), curr_dir, zeros.repeat(bsz, nb_tasks))
        alphas.append(alpha)
        dirs.append(curr_dir)

    # [bsz, len, tasks]
    return torch.stack(alphas, dim=1), torch.stack(dirs, dim=1)


def tac_forward_log(action_logprobs, stop_logprobs, lengths, subtask_lengths):
    """
    Args:
        action_logprobs: [bsz, len, tasks]
        stop_logprobs: [bsz, len, tasks,  2]
        lengths: [bsz]
        subtask_lengths [bsz]
    Returns:
        log_alphas [bsz, len, tasks]
    """
    bsz, length, nb_tasks = action_logprobs.shape[0], action_logprobs.shape[1], action_logprobs.shape[2]

    # Initial log alpha
    init_log_alpha = torch.ones([bsz, nb_tasks], device=action_logprobs.device) * NEG_INF
    init_log_alpha[:, 0] = action_logprobs[:, 0, 0]
    neg_inf = torch.tensor(NEG_INF, device=init_log_alpha.device)
    log_alphas = [init_log_alpha]
    for t in range(1, length):
        prev_log_alpha = log_alphas[-1] # [bsz, nb_task]
        lognonstop = prev_log_alpha + stop_logprobs[:, t, :, 0]
        logstop = prev_log_alpha + stop_logprobs[:, t, :, 1]
        logstop = torch.cat([neg_inf.repeat(bsz, 1), logstop[:, :-1]], dim=-1)
        loguntillnow = torch.logsumexp(torch.stack([logstop, lognonstop], dim=-1), dim=-1)
        log_alpha = action_logprobs[:, t, :] + loguntillnow
        subtask_mask = torch.arange(subtask_lengths.max(),
                                    device=subtask_lengths.device)[None, :] < subtask_lengths[:, None]
        log_alpha = log_alpha.masked_fill(~subtask_mask, NEG_INF)
        log_alpha = torch.where((t < lengths).unsqueeze(-1), log_alpha, neg_inf.repeat(bsz, nb_tasks))
        log_alphas.append(log_alpha)

    # [bsz, len, tasks]
    return torch.stack(log_alphas, dim=1)


def teacherforce_batch(modular_p: ModularPolicy, trajs: DictList, lengths,
                       dropout_p, decode=False):
    """ Return log probs of a trajectory """
    dropout_p = 0. if decode else dropout_p
    env_ids = trajs.env_id[:, 0]
    subtasks = []
    subtask_lengths = []
    unique_tasks = set()
    for env_id in env_ids:
        subtask = ID2SKETCHS[env_id]
        subtasks.append(torch.tensor(subtask, device=env_ids.device))
        subtask_lengths.append(len(subtask))
        for task_id in subtask:
            if not task_id in unique_tasks:
                unique_tasks.add(task_id)
    subtask_lengths = torch.tensor(subtask_lengths, device=env_ids.device)
    unique_tasks = list(unique_tasks)

    # Forward for all unique task
    # task_results [bsz, length, all_tasks]
    states = trajs.features.float()
    targets = trajs.action
    all_task_results = DictList()
    for task in unique_tasks:
        all_task_results.append(modular_p.forward(task, states, targets, dropout_p=dropout_p))
    all_task_results.apply(lambda _t: torch.stack(_t, dim=2))

    # pad subtasks
    subtasks = torch.nn.utils.rnn.pad_sequence(subtasks, batch_first=True, padding_value=0)

    # results [bsz, len, nb_tasks]
    results = DictList()
    for batch_id, subtask in enumerate(subtasks):
        curr_result = DictList()
        for task in subtask:
            task_id = unique_tasks.index(task)
            curr_result.append(all_task_results[batch_id, :, task_id])

        # [len, tasks]
        curr_result.apply(lambda _t: torch.stack(_t, dim=1))
        results.append(curr_result)
    results.apply(lambda _t: torch.stack(_t, dim=0))

    # Training
    if not decode:
        log_alphas = tac_forward_log(action_logprobs=results.action_logprobs,
                                     stop_logprobs=results.stop_logprobs,
                                     lengths=lengths,
                                     subtask_lengths=subtask_lengths)
        seq_logprobs = log_alphas[torch.arange(log_alphas.shape[0], device=log_alphas.device),
                                  lengths - 1, subtask_lengths - 1]
        avg_logprobs = seq_logprobs.sum() / lengths.sum()
        return {'loss': -avg_logprobs}

    # Decode
    else:
        alphas, _ = tac_forward(action_logprobs=results.action_logprobs,
                                stop_logprobs=results.stop_logprobs,
                                lengths=lengths, subtask_lengths=subtask_lengths)
        decoded = alphas.argmax(-1)
        batch_ids = torch.arange(decoded.shape[0], device=decoded.device).unsqueeze(-1).repeat(1, decoded.shape[1])
        decoded_subtasks = subtasks[batch_ids, decoded]
        total_task_corrects = 0
        for idx, (subtask, decoded_subtask, action, length) in enumerate(zip(subtasks, decoded_subtasks,
                                                                             trajs.action, lengths)):
            _decoded_subtask = decoded_subtask[:length]
            _action = action[:length]
            gt = get_subtask_seq(_action, subtask)
            total_task_corrects += (gt == _decoded_subtask).float().sum()
        return {'task_acc': total_task_corrects / lengths.sum()}, {'tru': gt, 'act': _action,
                                                                   'dec': _decoded_subtask}


def evaluate_on_env(modular_p: ModularPolicy, env, action_mode='greedy'):
    device = next(modular_p.parameters()).device
    modular_p.eval()
    modular_p.reset(env.env_id)
    obs = DictList(env.reset())
    obs.apply(lambda _t: torch.tensor(_t, device=device).float())
    done = False
    traj = DictList()
    while not done:
        action = modular_p.get_action(obs.features.unsqueeze(0), mode=action_mode)
        if action is not None:
            next_obs, reward, done, _ = env.step(action)
            transition = {'reward': reward, 'action': action,
                          'features': obs.features}
            traj.append(transition)

            obs = DictList(next_obs)
            obs.apply(lambda _t: torch.tensor(_t, device=device).float())
        else:
            done = True
    return traj
