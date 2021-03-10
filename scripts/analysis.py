"""
Run Omdec and analyze structure
"""
import cv2
from PIL import Image
import gym
import os
from gym.wrappers import TimeLimit
from gym_psketch.evaluate import f1
from gym_psketch.bots.omdec import OMdecBase
import gym_psketch.bots.demo_bot as demo
from gym_psketch import DictList, Actions
import argparse
import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt

from gym_psketch.visualize import idxpos2tree, visual, distance2ctree, tree_to_str

ACTION_VOCAB = ['↑', '↓', '←', '→', 'u', 'D']

parser = argparse.ArgumentParser()
parser.add_argument('--model_ckpt', required=True)
parser.add_argument('--envs', required=True, nargs='+')
parser.add_argument('--episodes', type=int, default=100)
parser.add_argument('--max_steps', type=int, default=80)
parser.add_argument('--width', type=int, default=10)
parser.add_argument('--height', type=int, default=10)
parser.add_argument('--outdir', default='out')
parser.add_argument('--use_demo', action='store_true')
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--plot_p', action='store_true')
parser.add_argument('--mp4', action='store_true')
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


def generate(bot, env, render_mode='ansi'):
    env_id = torch.tensor([env.env_id]).long()
    obs = DictList(env.reset())
    obs.apply(lambda _t: torch.tensor(_t).float().unsqueeze(0))
    mems = bot.init_memory(env_id)
    done = False
    traj = DictList()
    while not done:
        model_outputs = bot.get_action(obs, env_id, mems, 'greedy')
        viz = env.render(render_mode)
        next_obs, reward, done, _ = env.step(model_outputs.actions.item())
        traj.append(DictList({'action': model_outputs.actions,
                              'reward': torch.tensor(reward),
                              'viz': viz}))
        if isinstance(bot, OMdecBase):
            traj.append(DictList({'p': model_outputs.p.squeeze(0)}))
            if 'p_out' in model_outputs:
                traj.append(DictList({'p_out': model_outputs.p_out.squeeze(0)}))

        obs = DictList(next_obs)
        obs.apply(lambda _t: torch.tensor(_t).float().unsqueeze(0))
        mems = model_outputs.mems if bot.is_recurrent else None

    # Forcing outputting done action at the end, in order to plot the P.
    # So just do one more step
    if model_outputs.actions.item() != Actions.DONE.value:
        viz = env.render(render_mode)
        model_outputs = bot.get_action(obs, env_id, mems, 'greedy')
        traj.append(DictList({'action': model_outputs.actions,
                              'reward': torch.tensor(0),
                              'viz': viz}))

        if isinstance(bot, OMdecBase):
            traj.append(DictList({'p': model_outputs.p.squeeze(0)}))
            if 'p_out' in model_outputs:
                traj.append(DictList({'p_out': model_outputs.p_out.squeeze(0)}))

    success = env.state.sketch_id == len(env.sketchs)
    traj.apply(lambda _t: torch.stack(_t) if not isinstance(_t[0], str) else _t)
    return traj, success, traj.reward.sum()


def make_col(y,max_height, plot_char):
    #requires positive value
    col_list = []
    for i in range(max_height):
        if i >= y:
            col_list.append(' ')
        else:
            col_list.append(plot_char)
    return col_list


def make_neg_col(y,max_height, plot_char):
    #requires negative value
    col_list = []
    for i in reversed(range(max_height)):
        if y >= -i:
            col_list.append(' ')
        else:
            col_list.append(plot_char)
    return col_list


def scale_data(x, plot_height):
    '''scales list data to allow for floats'''
    result = []
    z = [abs(i) for i in x]
    for i in x:
        temp = i/float(max(z))
        temp2 = temp*plot_height
        result.append(int(temp2))
    return result


def plot(x, actions, plot_height=10, plot_char=u'\u25cf'):
    ''' takes a list of ints or floats x and makes a simple terminal histogram.
        This function will make an inaccurate plot if the length of data list is larger than the number of columns
        in the terminal.'''
    lines = []
    x = scale_data(x, plot_height)

    max_pos = max(x)
    max_neg = abs(min(x))

    hist_array = []
    neg_array = []

    for i in x:
        hist_array.append(make_col(i, max_pos, plot_char))

    for i in x:
        neg_array.append(make_neg_col(i, max_neg, plot_char))

    for i in range(len(neg_array)):
        neg_array[i].extend(hist_array[i])

    for i in reversed(range(len(neg_array[0]))):
        line = []
        empty_line = True
        for j in range(len(neg_array)):
            line.append(neg_array[j][i])
            if neg_array[j][i] != " ":
                empty_line = False

        # Skip empty line
        if not empty_line:
            lines.append(' '.join(line))
    lines.append(" ".join([ACTION_VOCAB[a.item()] for a in actions]))
    return '\n'.join(lines)


def get_p_plot(actions, ps, time_step=None, info_line=None):
    nb_slots = ps.shape[1] - 1
    p_avg_fig, p_avg_ax = plt.subplots()
    ps = ps[:-1, 1:]
    ps = (ps / ps.sum(1, keepdim=True)).transpose(1, 0)
    action_ticks = [ACTION_VOCAB[actions[t].item()] for t in range(len(actions) - 1)]
    p_avg_ax.imshow(ps, cmap='Greys')
    p_avg_ax.set_xticks([i for i in range(len(action_ticks))])
    p_avg_ax.set_xticklabels(action_ticks)
    p_avg_ax.set_yticks([i for i in range(nb_slots)])
    p_avg_ax.set_yticklabels(['slot{}'.format(nb_slots - i) for i in range(nb_slots)])

    if time_step is not None:
        highlight_x = time_step - 0.5
        p_avg_ax.plot([highlight_x, highlight_x], [-0.5, nb_slots - 1 + 0.5], 'r')

    if info_line is not None:
        p_avg_ax.set_xlabel(info_line, fontsize=15)
    return p_avg_fig


def process_env(args, bot, env_name):
    succs = []
    rets = []
    lines = []
    parsing_metric = DictList()
    model_name = os.path.dirname(os.path.abspath(args.model_ckpt)).split('/')[-1] + '_{}'.format(env_name)
    if args.use_demo:
        model_name = model_name + '_demo'
    render_mode = 'rgb' if args.mp4 else 'ansi'
    for episode_id in tqdm.tqdm(range(args.episodes)):
        env = TimeLimit(gym.make(env_name,
                                 height=args.height,
                                 width=args.width), args.max_steps)
        if args.seed is not None:
            env.seed(args.seed + episode_id)
        if args.use_demo:
            demo_bot = demo.DemoBot(env=env)
            while True:
                try:
                    ret, _demo_traj, viz = demo.generate_one_traj(demo_bot, env,
                                                                  render_mode=render_mode)
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
            success = True
        else:
            with torch.no_grad():
                traj, success, ret = generate(bot, env)
                ret = ret.item()

        succs.append(float(success))
        rets.append(ret)
        lines.append('########## [{}]EPISODE {} ##############'.format(env_name, episode_id))
        lines.append('return: {}  success: {}'.format(ret, success))

        has_p = 'p' in traj
        states = traj.viz
        rewards = traj.reward
        actions = traj.action
        if has_p:
            ps = traj.p
            ps[0, :-1] = 0
            ps[0, -1] = 1
            lines.append('P: ')
            lines.append(idxpos2tree(actions, ps))
            if 'p_out' in traj:
                lines.append('P_out:')
                lines.append(idxpos2tree(actions, traj.p_out))
            slots = torch.arange(len(ps[0]))
            p_avg = len(slots) - (slots[None, :] * ps).sum(-1)
            p_avg_str = plot(p_avg, actions=actions)
            lines.append('p_avg:')
            lines.append(p_avg_str)

            lines.append('Tree')
            depths = p_avg[:-1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            depths = np.digitize(depths.numpy(), bins=np.linspace(0, 1, 5))
            parse_tree = distance2ctree(depths, [ACTION_VOCAB[a.item()] for a in actions], False)
            tree_line = tree_to_str(parse_tree)
            lines.append(tree_line[1:-1])

            # Compute F1
            if args.use_demo:
                use_ids = (traj.action.reshape(-1)[1:-1] == Actions.USE.value).nonzero().view(-1).cpu().numpy()
                target = use_ids.tolist()
                p_vals = torch.arange(bot.nb_slots + 1)
                avg_p = (p_vals * ps[1:-1]).sum(-1)
                for k in [3, 4, 5, 6]:
                    _, inds = (-avg_p).topk(k)
                    preds = inds.tolist()
                    for tol in [0]:
                        result = f1(target, preds, tol, with_detail=True)
                        for name in result:
                            parsing_metric.append({'{}_tol{}_k{}'.format(name, tol, k): result[name]})

            # Generate Plots
            if args.plot_p:
                p_avg_fig = get_p_plot(actions, ps)
                p_avg_fig.savefig(os.path.join(args.outdir, model_name + '_{}.png'.format(episode_id)),
                                  bbox_inches='tight')
                plt.close(p_avg_fig)

        # Save episode details
        if not args.mp4:
            episode_lines = []
            for t in range(len(rewards)):
                episode_lines.append('################################')
                episode_lines.append('Sketch: {}'.format(env.sketchs))
                info_line = "steps: {}\taction: {}\treward: {}".format(
                    t, ACTION_VOCAB[actions[t].item()], rewards[t])
                if has_p:
                    info_line += '\tp: {}'.format(np.array2string(ps[t].clamp(min=1e-8).numpy(),
                                                                  formatter={'float_kind': lambda x: visual(x, 1)}))
                episode_lines.append(info_line)
                episode_lines.append(states[t])

            episode_res_name = os.path.join(args.outdir, model_name + '_{}.txt'.format(episode_id))
            with open(episode_res_name, 'w') as f:
                f.write('\n'.join(episode_lines))

        # Save MP4
        else:
            returns = rewards.cumsum(0)
            if not has_p:
                frames = states
            else:
                frames = []
                sketch_id = 0
                prev_ret = 0
                for time_step, state_frame in enumerate(states):
                    curr_ret = returns[time_step - 1].item() if time_step - 1 > 0 else 0
                    curr_rwd = rewards[time_step - 1].item() if time_step - 1 > 0 else 0
                    info_line = "steps: {}, reward: {}, ret: {} \n subtask: {}".format(
                        time_step, curr_rwd, curr_ret, env.sketchs[sketch_id])
                    special = curr_ret > prev_ret
                    if special:
                        sketch_id += 1
                        prev_ret = returns[time_step - 1]
                        info_line += "(done)"
                    p_fig = get_p_plot(actions, ps, time_step, info_line)
                    w, h = p_fig.canvas.get_width_height()
                    p_fig.tight_layout(pad=0)
                    p_fig.canvas.draw()
                    p_img = np.fromstring(p_fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
                    plt.close(p_fig)

                    # Concat
                    final_frame = np.concatenate([p_img, state_frame], axis=1)
                    frames.append(final_frame)

                    if special:
                        for _ in range(2):
                            frames.append(final_frame)

            # Write to mp4
            print('Producing videos...')
            frames = [Image.fromarray(frame) for frame in frames]

            # Repeat ending frame for additional time
            frames.append(frames[-1])
            videodims = (frames[0].width, frames[0].height)
            video = cv2.VideoWriter(os.path.join(args.outdir, model_name + "_{}.mp4".format(episode_id)),
                                    0x7634706d, 1, videodims)
            for frame in frames:
                video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
            video.release()

    lines.append('######################{}#########################'.format(env_name))
    lines.append('Avg return {}'.format(sum(rets) / args.episodes))
    lines.append('Avg success rate {}'.format(sum(succs) / args.episodes))
    print('{} return: {} success {}'.format(env_name, np.mean(rets), np.mean(succs)))
    parsing_metric.apply(lambda _t: np.mean(_t))
    for key, val in parsing_metric.items():
        print(key, val)
    with open(os.path.join(args.outdir, model_name + '.out'), 'w') as f:
        f.write('\n'.join(lines))


def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    bot = torch.load(args.model_ckpt, map_location=torch.device('cpu'))
    bot.eval()

    for env_name in args.envs:
        process_env(args, bot, env_name)


if __name__ == '__main__':
    main(args)
