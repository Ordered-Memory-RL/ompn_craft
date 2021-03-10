"""
Visuzlize a bot on a env
"""
import gym
from gym_psketch.visualize import visual
import gym_psketch
from gym.wrappers import TimeLimit
import curses
import argparse
import torch
from typing import Dict
import numpy


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', choices=['demo', 'keyboard', 'model'], default='keyboard')
    parser.add_argument('-max_steps', type=int, default=40, help='Maximum steps')
    parser.add_argument('-model_ckpt', default=None)
    parser.add_argument('-env', default='makeplank-v0')
    parser.add_argument('-debug', action='store_true')
    return parser.parse_args()


def show(env, window, info: Dict):
    window.clear()
    window.addstr(0, 0, 'Instructions: ' + env.instructions)
    window.addstr(1, 0, 'Sketchs: ' + str(env.sketchs))
    info_line = ['{}: {}'.format(k, v) for k, v in info.items()]
    window.addstr(2, 0, '   '.join(info_line))
    window.addstr(4, 0, env.render('ansi'))
    window.refresh()


class Renderer:
    def __init__(self, args):
        self.env = TimeLimit(gym.make(args.env), max_episode_steps=args.max_steps)

    def get_action(self, obs, ch):
        raise NotImplementedError

    def reset(self, init_obs):
        pass

    def main_loop(self, window):
        obs = self.env.reset()
        self.reset(obs)
        done = False
        action = None
        reward = None
        steps = 0
        ret = 0
        while not done:
            self.display(action, done, ret, reward, steps, window)
            ch = window.getch()
            action = self.get_action(obs, ch)
            obs, reward, done, _ = self.env.step(action)
            ret += reward
            steps += 1

        # Clear screen
        self.display(action, done, ret, reward, steps, window)
        window.getch()

    def display(self, action, done, ret, reward, steps, window):
        show(self.env, window, {'steps': steps,
                                'action': gym_psketch.ID2ACTIONS[action] if action is not None else action,
                                'reward': reward,
                                'return': ret,
                                'done': done})


class Model(Renderer):
    def __init__(self, args):
        super(Model, self).__init__(args)
        with open(args.model_ckpt, 'rb') as f:
            self.bot = torch.load(f, map_location=torch.device('cpu'))
        self.bot.eval()
        self.mems = None
        self.output = None

    def reset(self):
        with torch.no_grad():
            self.mems = self.bot.init_memory(torch.tensor([self.env.env_id]).long())
        self.output = None

    def get_action(self, obs, ch):
        obs = torch.tensor(obs).float()
        with torch.no_grad():
            output = self.bot.get_action(obs.unsqueeze(0), self.mems)
        self.output = output
        if self.bot.is_recurrent:
            self.mems = output.mems
        return output.actions.squeeze(0).item()

    def display(self, action, done, ret, reward, steps, window):
        p = 'none'
        p_avg = 'none'
        if self.output is not None:
            p = self.output.p.squeeze(0).cpu().numpy()
            p_avg = ((len(p) - numpy.arange(len(p))) * p).sum()
            p = numpy.array2string(p, formatter={'float_kind': lambda x: visual(x, 1)})
        show(self.env, window, {'steps': steps,
                                'action': gym_psketch.ID2ACTIONS[action] if action is not None else action,
                                'reward': reward,
                                'return': ret,
                                'done': done,
                                'p': p,
                                'p_avg': p_avg})


class Keyboard(Renderer):

    def get_action(self, obs, ch):
        if ch == curses.KEY_UP:
             action = gym_psketch.Actions.UP.value
        elif ch == curses.KEY_DOWN:
            action = gym_psketch.Actions.DOWN.value
        elif ch == curses.KEY_LEFT:
            action = gym_psketch.Actions.LEFT.value
        elif ch == curses.KEY_RIGHT:
            action = gym_psketch.Actions.RIGHT.value
        elif ch == ord('u'):
            action = gym_psketch.Actions.USE.value
        else:
            action = gym_psketch.Actions.DONE.value
        return action


class Demo(Renderer):
    def __init__(self, args):
        super(Demo, self).__init__(args)
        self.bot = gym_psketch.DemoBot(self.env)

    def reset(self, init_obs):
        self.bot.reset()

    def get_action(self, obs, ch):
        return self.bot.get_action(obs)

    def display(self, action, done, ret, reward, steps, window):
        stack = '\n' + self.bot.stack.__repr__()
        show(self.env, window, {'steps': steps,
                                'action': gym_psketch.ID2ACTIONS[action] if action is not None else action,
                                'reward': reward,
                                'return': ret,
                                'done': done,
                                'stack': stack})


def run(args):
    if args.mode == 'keyboard':
        renderer = Keyboard(args)
    elif args.mode == 'demo':
        renderer = Demo(args)
    elif args.mode == 'model':
        renderer = Model(args)
    else:
        raise ValueError
    while True:
        curses.wrapper(renderer.main_loop)


if __name__ == '__main__':
    run(get_args())
