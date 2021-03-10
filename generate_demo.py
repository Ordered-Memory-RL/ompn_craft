import gym_psketch
from gym_psketch.bots.demo_bot import PlanningError, generate_one_traj
from absl import flags
import gym
from gym.wrappers import TimeLimit
import numpy as np
import pickle
import torch
import os
from gym_psketch import DictList
from gym_psketch.evaluate import batch_evaluate

FLAGS = flags.FLAGS


flags.DEFINE_integer('demo_episodes', default=500, help='Number of episodes generated')
flags.DEFINE_string('demo_ckpt', default=None, help='Path to demo checkpoint')


def generate_from_ckpt(env_fn):
    bot = torch.load(FLAGS.demo_ckpt, map_location=torch.device('cpu'))
    bot.eval()
    trajs = []
    rets = []
    succs = []
    batch_size = 512
    while len(trajs) < FLAGS.demo_episodes:
        envs = [env_fn() for _ in range(batch_size)]
        batch_trajs = batch_evaluate(envs=envs, bot=bot, action_mode='greedy')
        for traj in batch_trajs:
            ret = traj.rewards.sum().item()
            succ = ret >= len(envs[0].sketchs)
            succs.append(succ)
            rets.append(ret)

            # Append successful trajs
            if succ:
                # make it numpy list
                _traj = {}
                for name, values in traj.items():
                    new_values = values.split(1)
                    new_values = [v.squeeze(0).numpy() if len(v.shape) > 1 else v.item()
                                  for v in new_values]
                    _traj[name] = new_values
                _traj['action'] = _traj['actions']
                del _traj['actions']
                trajs.append(_traj)

        print('Generate {} trajs, succs {} rets'.format(len(trajs), np.mean(succs), np.mean(rets)))
    return trajs[:FLAGS.demo_episodes]


def generate_env_demo(env_fn):
    env = env_fn()
    bot = gym_psketch.DemoBot(env)
    num_episodes = 0
    trajs = []
    rets = []
    succs = []
    while len(trajs) < FLAGS.demo_episodes:
        try:
            ret, traj = generate_one_traj(bot, env, render_mode=None)
            succ = ret >= len(env.sketchs)
            rets.append(ret)
            succs.append(succ)
            if succ:
                trajs.append(traj)
            num_episodes += 1
            if num_episodes % 100 == 0:
                print('Generating 100 episodes...sucess {} ret {}'.format(np.mean(succs), np.mean(rets)))

        except PlanningError:
            print('Warning: Non achievable, re-generate this traj')
            succs.append(False)
            pass
    print('Success rate: {}'.format(np.mean(succs)))
    print('Avg. return: {}'.format(np.mean(rets)))
    return trajs


def main():
    for env_name in FLAGS.envs:
        print('Generating {}'.format(env_name))
        env_fn = lambda : TimeLimit(gym.make(env_name, width=FLAGS.width, height=FLAGS.height),
                                    max_episode_steps=FLAGS.max_steps)
        env = env_fn()
        if FLAGS.demo_ckpt is None:
            trajs = generate_env_demo(env_fn)
        else:
            trajs = generate_from_ckpt(env_fn)
        lengths = [traj['action'].__len__() for traj in trajs]
        avg_lengths = sum(lengths) / float(len(lengths))
        print('Saving... with avg length {}'.format(avg_lengths))
        if FLAGS.demo_ckpt is not None:
            demo_name = os.path.join(gym_psketch.DATASET_DIR, '{}_bot.pkl'.format(env_name))
        else:
            demo_name = os.path.join(gym_psketch.DATASET_DIR, '{}.pkl'.format(env_name))
        with open(demo_name, 'wb') as f:
            pickle.dump({'trajs': trajs, 'env_id': env.env_id}, f)

