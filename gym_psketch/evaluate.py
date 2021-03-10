"""
Batch evaluation on env
"""
import gym
from gym_psketch import Actions, ID2SKETCHS, ACTION_VOCAB
from gym.wrappers import TimeLimit
from gym_psketch.dataloader import Dataloader
from gym_psketch.bots import ModelBot
from gym_psketch.utils import DictList
import numpy as np
import random
from gym_psketch.visualize import idxpos2tree
import torch
from typing import List
from absl import flags
FLAGS = flags.FLAGS


__all__ = ['batch_evaluate', 'evaluate_loop']


def step_batch_envs(envs, actions, actives):
    """ Step a batch of envs. And detect if there are inactive/done envs
    return obss, rewards, dones of the active envs
    """
    assert actions.shape[0] == len(actives)
    active_envs = [envs[i] for i in actives]
    obss = DictList()
    rewards = []
    dones = []
    for action, env in zip(actions, active_envs):
        obs, reward, done, _ = env.step(action.item())
        obss.append(obs)
        rewards.append(reward)
        dones.append(done)

    obss.apply(lambda _t: torch.tensor(_t).float())
    rewards = torch.tensor(rewards).float()
    dones = torch.tensor(dones)

    if FLAGS.cuda:
        obss.apply(lambda _t: _t.cuda())
        rewards = rewards.cuda()
        dones = dones.cuda()

    # Update active
    return obss, rewards, dones


def batch_evaluate(envs: List, bot: ModelBot, action_mode) -> List:
    """ Return trajectories after roll out """
    obs = DictList()
    for env in envs:
        obs.append(DictList(env.reset()))
    obs.apply(lambda _t: torch.tensor(_t).float())
    actives = torch.tensor([i for i in range(len(envs))])
    if FLAGS.cuda:
        obs.apply(lambda _t: _t.cuda())
        actives = actives.cuda()

    trajs = [DictList() for _ in range(len(envs))]
    env_ids = torch.tensor([env.env_id for env in envs]).long().to(device=obs.img.device)
    mems = bot.init_memory(env_ids) if bot.is_recurrent else None

    # Continue roll out while at least one active
    steps = 0
    while len(actives) > 0:
        active_trajs = [trajs[i] for i in actives]
        with torch.no_grad():
            model_outputs = bot.get_action(obs, env_ids[actives], mems, mode=action_mode)
        actions = model_outputs.actions
        next_obs, rewards, dones = step_batch_envs(envs, actions, actives)
        transition = DictList({'rewards': rewards, 'actions': actions})
        transition.update(obs)

        for idx, active_traj in enumerate(active_trajs):
            active_traj.append(transition[idx])
        steps += 1

        # Memory
        next_mems = None
        if bot.is_recurrent:
            next_mems = model_outputs.mems

        # For next step
        un_done_ids = (~dones).nonzero().squeeze(-1)
        obs = next_obs[un_done_ids]
        actives = actives[un_done_ids]
        mems = next_mems[un_done_ids] if next_mems is not None else None

    for traj in trajs:
        traj.apply(lambda _tensors: torch.stack(_tensors))
    return trajs


def evaluate_loop(bot, val_metrics, action_mode='greedy'):
    for env_name in val_metrics:
        envs = [TimeLimit(gym.make(env_name, width=FLAGS.width, height=FLAGS.height),
                          max_episode_steps=FLAGS.max_steps)
                for _ in range(FLAGS.eval_episodes)]
        with torch.no_grad():
            trajs = batch_evaluate(envs=envs, bot=bot, action_mode=action_mode)
            returns = []
            lengths = []
            succs = []
            for traj, env in zip(trajs, envs):
                ret = traj.rewards.sum()
                succs.append((ret >= len(env.sketchs)).float())
                returns.append(ret)
                lengths.append(traj.rewards.__len__())
        val_metrics[env_name]['ret'] = sum(returns) / float(FLAGS.eval_episodes)
        val_metrics[env_name]['succs'] = sum(succs) / float(FLAGS.eval_episodes)
        val_metrics[env_name]['episode_length'] = sum(lengths) / float(FLAGS.eval_episodes)
    return val_metrics


def get_use_ids(actions, env_name):
    use_ids = (actions.reshape(-1) == Actions.USE.value).nonzero().view(-1).cpu().numpy()
    if env_name is not None:
        if 'getgem' in env_name:
            use_ids = np.array(use_ids[:4].tolist() +
                               use_ids[-1:].tolist())
        elif 'getgold' in env_name:
            use_ids = np.array(use_ids[:3].tolist() +
                               use_ids[-1:].tolist())
    return use_ids


def parsing_loop(bot, dataloader: Dataloader):
    bot.eval()
    parsing_metric = {env: DictList() for env in dataloader.env_names}
    all_trajs = dataloader.data['val']
    for env_name in dataloader.env_names:
        trajs = all_trajs[env_name]
        random.shuffle(trajs)
        total_task_corrects = 0
        total_lengths = 0
        for traj in trajs[:FLAGS.eval_episodes]:
            demo_traj = DictList(traj)

            # Remove done
            demo_traj.apply(lambda _t: torch.tensor(_t)[:-1].unsqueeze(0))
            if FLAGS.cuda:
                demo_traj.apply(lambda _t: _t.cuda())

            length = demo_traj.action.shape[1]
            env_ids = demo_traj.env_id[:, 0]
            mems = bot.init_memory(env_ids)
            result = DictList()
            for t in range(length):
                curr_transition = demo_traj[:, t]
                with torch.no_grad():
                    model_output = bot.forward(obs=curr_transition,
                                               mems=mems,
                                               env_ids=curr_transition.env_id)
                    output = {'action': curr_transition.action,
                              'p': model_output.p.squeeze(0),
                              'reward': curr_transition.reward}
                    result.append(output)
                    mems = model_output.mems
            result.apply(lambda _t: torch.stack(_t))
            use_ids = get_use_ids(demo_traj.action, env_name)
            target = use_ids.tolist()

            # Get prediction sorted
            ps = result.p.cpu()
            ps[0, :-1] = 0
            ps[0, -1] = 1
            p_vals = torch.arange(bot.nb_slots + 1)
            avg_p = (p_vals * ps).sum(-1)
            _, inds = (-avg_p).topk(len(use_ids))
            inds = inds[inds.sort()[1]]

            # F1 score
            preds = inds.tolist()
            parsing_metric[env_name].append({'f1_tol{}'.format(tol): f1(target, preds, tol) for tol in [0, 1]})

            # Compute task accuracy
            _action = demo_traj.action.reshape(-1)

            # Sort it to have normal task decoding
            _decoded_subtask = get_subtask_seq(_action, subtask=ID2SKETCHS[env_ids.item()],
                                               use_ids=inds)
            _gt_subtask = get_subtask_seq(_action,
                                          subtask=ID2SKETCHS[env_ids.item()],
                                         use_ids=get_use_ids(_action, env_name))
            total_task_corrects += (_gt_subtask == _decoded_subtask).float().sum().item()
            total_lengths += length

            for threshold in [0.4, 0.5, 0.6]:
                preds = get_boundaries(ps, bot.nb_slots, threshold=threshold, nb_boundaries=len(target))
                parsing_metric[env_name].append({'f1_tol{}_thres{}'.format(tol, threshold): f1(target, preds, tol)
                                                 for tol in [0]})
                _decoded_subtask = get_subtask_seq(_action, subtask=ID2SKETCHS[env_ids.item()],
                                                   use_ids=np.array(preds))
                _corrects = (_decoded_subtask == _gt_subtask).float()
                for _c in _corrects:
                    parsing_metric[env_name].append({'task_acc_thres{}'.format(threshold): _c.item()})

        parsing_metric[env_name].task_acc = total_task_corrects / total_lengths

    # Visualize last episode
    lines = []
    actions = demo_traj.action.reshape(-1)
    ps = result.p
    lines.append('P: ')
    lines.append(idxpos2tree(actions, ps))

    # print task alignment
    lines += get_ta_lines(action=_action, decoded_subtask=_decoded_subtask,
                          gt_subtask=_gt_subtask)
    lines.append('use_ids: {}'.format(use_ids.tolist()))
    lines.append('dec_ids: {}'.format(inds.tolist()))
    return parsing_metric, lines


def f1(targets, preds, tolerance=1, with_detail=False):
    nb_preds = len(preds)
    nb_targets = len(targets)
    correct_targets = 0
    for tar in targets:
        matched = False
        for pred in preds:
            if tar <= pred + tolerance and tar >= pred - tolerance:
                matched = True
                break

        if matched:
            correct_targets += 1
    recall = correct_targets / nb_targets

    correct_preds = 0
    for pred in preds:
        matched = False
        for tar in targets:
            if pred <= tar + tolerance and pred >= tar - tolerance:
                matched = True
                break
        if matched:
            correct_preds += 1
    pre = correct_preds / nb_preds
    f1 = 2 * pre * recall / (pre + recall) if pre + recall > 0 else 0.
    if with_detail:
        return {'f1': f1, 'pre': pre, 'recall': recall}
    else:
        return f1


def get_subtask_seq(action, subtask, use_ids):
    task_end_ids = use_ids + 1
    start = 0
    gt = torch.zeros_like(action)
    for i, end_id in enumerate(task_end_ids):
        gt[start: end_id] = subtask[i]
        start = end_id
    return gt


def get_ta_lines(action, decoded_subtask, gt_subtask):
    lines = ['Task Alignment']
    if isinstance(action, torch.Tensor):
        action = action.tolist()
    if isinstance(decoded_subtask, torch.Tensor):
        decoded_subtask = decoded_subtask.tolist()
    if isinstance(gt_subtask, torch.Tensor):
        gt_subtask = gt_subtask.tolist()

    lines.append('act:' + ' '.join([ACTION_VOCAB[a] for a in action]))
    lines.append('tru:' + ' '.join([str(e) for e in gt_subtask]))
    lines.append('dec:' + ' '.join([str(e) for e in decoded_subtask]))
    return lines


def get_boundaries(ps, nb_slots, threshold, nb_boundaries=None):
    """ Pick the end of segment as boundaries """
    # Get average p standardized
    ps[0, :-1] = 0
    ps[0, -1] = 1
    p_vals = torch.arange(nb_slots + 1, device=ps.device).flip(0)
    avg_p = (p_vals * ps).sum(-1)
    avg_p = avg_p / (avg_p.max() - avg_p.min())

    # Predicted boundaries as end of segment
    preds = []
    prev = False
    for t in range(len(avg_p)):
        curr = avg_p[t] > threshold
        if not prev and not curr:
            pass
        elif prev and curr:
            pass
        else:
            if prev and not curr:
                preds.append(t - 1)
            prev = curr
    preds.append(len(avg_p) - 1)
    if nb_boundaries is not None:
        preds = preds[::-1][:nb_boundaries][::-1]
    return preds

