"""
Run Omdec and analyze structure
"""
import gym
import os
from gym.wrappers import TimeLimit
from gym_psketch.evaluate import f1, get_boundaries
from gym_psketch.bots.omdec import OMdecBase
from gym_psketch.bots.bots_iclr.model_bot import OMDecCompBot as ICLROmdec
import gym_psketch.bots.demo_bot as demo
from gym_psketch import DictList, Actions, ACTION_VOCAB
import argparse
import random
import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt

from gym_psketch.visualize import idxpos2tree, visual, distance2ctree, tree_to_str

parser = argparse.ArgumentParser()
parser.add_argument('--model_prefix', required=True)
parser.add_argument('--runs', default=5, type=int)
parser.add_argument('--envs', required=True, nargs='+')
parser.add_argument('--episodes', type=int, default=50)
parser.add_argument('--outdir', default='out')
parser.add_argument('--seed', default=None, type=int)
args = parser.parse_args()


def teacher_force(bot, demo_traj):
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
                      'reward': curr_transition.reward}
            if 'p' in model_output:
                output.update({'p': model_output.p.squeeze(0)})
            result.append(output)
            mems = model_output.mems
    result.apply(lambda _t: torch.stack(_t))
    return result


def generate(bot, env):
    env_id = torch.tensor([env.env_id]).long()
    obs = DictList(env.reset())
    obs.apply(lambda _t: torch.tensor(_t).float().unsqueeze(0))
    mems = bot.init_memory(env_id)
    done = False
    traj = DictList()
    while not done:
        model_outputs = bot.get_action(obs, env_id, mems, 'greedy')
        viz = env.render('ansi')
        next_obs, reward, done, _ = env.step(model_outputs.actions.item())
        traj.append(DictList({'action': model_outputs.actions,
                              'reward': torch.tensor(reward),
                              'viz': viz}))
        if isinstance(bot, OMdecBase) or isinstance(bot, ICLROmdec):
            traj.append(DictList({'p': model_outputs.p.squeeze(0)}))
            if 'p_out' in model_outputs:
                traj.append(DictList({'p_out': model_outputs.p_out.squeeze(0)}))

        obs = DictList(next_obs)
        obs.apply(lambda _t: torch.tensor(_t).float().unsqueeze(0))
        mems = model_outputs.mems if bot.is_recurrent else None

    # Forcing outputting done action at the end, in order to plot the P.
    # So just do one more step
    if model_outputs.actions.item() != Actions.DONE.value:
        viz = env.render('ansi')
        model_outputs = bot.get_action(obs, env_id, mems, 'greedy')
        traj.append(DictList({'action': model_outputs.actions,
                              'reward': torch.tensor(0),
                              'viz': viz}))

        if isinstance(bot, OMdecBase) or isinstance(bot, ICLROmdec):
            traj.append(DictList({'p': model_outputs.p.squeeze(0)}))
            if 'p_out' in model_outputs:
                traj.append(DictList({'p_out': model_outputs.p_out.squeeze(0)}))

    success = env.state.sketch_id == len(env.sketchs)
    traj.apply(lambda _t: torch.stack(_t) if not isinstance(_t[0], str) else _t)
    return traj, success, traj.reward.sum()


def process_env(args, bot):
    parsing_metric = DictList()
    for episode_id in tqdm.tqdm(range(args.episodes)):
        env = TimeLimit(gym.make(random.choice(args.envs)), 100)
        if args.seed is not None:
            env.seed(args.seed + episode_id)

        demo_bot = demo.DemoBot(env=env)
        while True:
            try:
                ret, _demo_traj, viz = demo.generate_one_traj(demo_bot, env,
                                                              render_mode='ansi')
                if ret < len(env.sketchs):
                    continue
                demo_traj = DictList(_demo_traj)
                demo_traj.done = [False] * (len(demo_traj) - 1) + [True]
                demo_traj.action = [a.value for a in _demo_traj['action']]
                demo_traj.env_id = [env.env_id] * len(demo_traj)
                demo_traj.apply(lambda _t: torch.tensor(_t).unsqueeze(0) if not isinstance(_t[0], str) else _t)
                break
            except demo.PlanningError:
                pass
        with torch.no_grad():
            traj = teacher_force(bot, demo_traj)
            traj.viz = viz

        ps = traj.p
        ps[0, :-1] = 0
        ps[0, -1] = 1

        # Compute F1
        use_ids = (traj.action.reshape(-1)[1:-1] == Actions.USE.value).nonzero().view(-1).cpu().numpy()
        target = use_ids.tolist()
        p_vals = torch.arange(bot.nb_slots + 1)
        avg_p = (p_vals * ps[1:-1]).sum(-1)
        for k in [2, 3, 4, 5, 6]:
            _, inds = (-avg_p).topk(k)
            preds = inds.tolist()
            for tol in [1]:
                result = f1(target, preds, tol, with_detail=True)
                for name in result:
                    parsing_metric.append({'{}_tol{}_k{}'.format(name, tol, k): result[name]})

    parsing_metric.apply(lambda _t: np.mean(_t))
    return parsing_metric


def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    stats = []
    for _run in range(args.runs):
        run = _run + 1
        model_ckpt = args.model_prefix + "{}/bot_best.pkl".format(run)
        bot = torch.load(model_ckpt, map_location=torch.device('cpu'))
        bot.eval()
        parsing_metric = process_env(args, bot)
        stats.append(parsing_metric)
    columns = list(parsing_metric.keys())
    data = [[metric.get(col) for col in columns] for metric in stats]
    mean = np.mean(data, 0)
    std = np.std(data, 0)
    from pandas import DataFrame
    df = DataFrame([mean, std], index=['mean', 'std'], columns=columns)
    print(df.to_string())



if __name__ == '__main__':
    main(args)
