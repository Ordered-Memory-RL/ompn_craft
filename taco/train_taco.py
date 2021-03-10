from taco.model import ModularPolicy
from gym_psketch.env import Actions
from gym_psketch import DictList, NB_SUBTASKS
from gym_psketch.evaluate import get_ta_lines
from taco.data import Dataloader
from taco.core import teacherforce_batch, evaluate_on_env
import gym
import os
import time
from gym.wrappers import TimeLimit
from tensorboardX import SummaryWriter
import numpy as np
import torch
from absl import logging, flags
FLAGS = flags.FLAGS
flags.DEFINE_float('taco_lr', default=0.001, help='taco learning rate')
flags.DEFINE_integer('taco_batch_size', default=16, help='taco batch size')
flags.DEFINE_integer('taco_train_steps', default=4000, help='taco train steps')
flags.DEFINE_integer('taco_eval_freq', default=10, help='taco eval freq')
flags.DEFINE_float('taco_dropout', default=0.7, help='Initial dropout rate')
flags.DEFINE_float('taco_do_decay_ratio', default=0.8, help='use ratio * train_steps to decay')


class DropoutScheduler:
    def __init__(self):
        self.init_do = FLAGS.taco_dropout
        self.decay_steps = int(FLAGS.taco_do_decay_ratio * FLAGS.taco_train_steps)
        self.min_val = 0.
        self.steps = 0

    @property
    def dropout_p(self):
        return max((self.decay_steps - self.steps) / self.decay_steps * self.init_do, self.min_val)

    def step(self):
        self.steps += 1


def evaluate_loop(dataloader, model, dropout_p):
    # Testing
    val_metrics = {}
    model.eval()
    for env_name in dataloader.env_names:
        val_metrics[env_name] = DictList()

        # Teacher Forcing
        val_iter = dataloader.val_iter(batch_size=FLAGS.taco_batch_size, env_names=[env_name])
        for val_batch, val_lengths in val_iter:
            if FLAGS.cuda:
                val_batch.apply(lambda _t: _t.cuda())
                val_lengths = val_lengths.cuda()
            with torch.no_grad():
                batch_res = teacherforce_batch(model, trajs=val_batch, lengths=val_lengths,
                                               decode=False, dropout_p=dropout_p)
            val_metrics[env_name].append(batch_res)

        # parsing
        val_iter = dataloader.val_iter(batch_size=FLAGS.eval_episodes, env_names=[env_name])
        val_batch, val_lengths = val_iter.__next__()
        if FLAGS.cuda:
            val_batch.apply(lambda _t: _t.cuda())
            val_lengths = val_lengths.cuda()
        with torch.no_grad():
            parsing_res, parsing_info = teacherforce_batch(model, trajs=val_batch, lengths=val_lengths,
                                                           dropout_p=dropout_p, decode=True)
        val_metrics[env_name].append(parsing_res)

        # Print parsing info
        parsing_lines = get_ta_lines(action=parsing_info['act'],
                                     decoded_subtask=parsing_info['dec'],
                                     gt_subtask=parsing_info['tru'])
        logging.info('\n'.join(parsing_lines))

        # Free Run
        env = TimeLimit(gym.make(env_name, width=FLAGS.width, height=FLAGS.height),
                        max_episode_steps=FLAGS.max_steps)
        free_run_metrics = DictList()
        for _ in range(FLAGS.eval_episodes):
            with torch.no_grad():
                _traj = evaluate_on_env(modular_p=model, env=env, action_mode='greedy')
            if 'reward' in _traj:
                free_run_metrics.append({
                    'ret': np.sum(_traj.reward),
                    'succs': np.sum(_traj.reward) >= len(env.sketchs),
                    'episode_length': len(_traj.reward)
                })
        val_metrics[env_name].update(free_run_metrics)

        # Mean everything
        val_metrics[env_name].apply(lambda _t: torch.tensor(_t).float().mean())
    return val_metrics


def main(training_folder):
    logging.info('start taco...')
    first_env = gym.make(FLAGS.envs[0])
    n_feature = first_env.n_features
    model = ModularPolicy(nb_subtasks=NB_SUBTASKS, input_dim=n_feature,
                          n_actions=len(Actions) - 1)
    if FLAGS.cuda:
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.taco_lr)

    dataloader = Dataloader(FLAGS.envs, 0.2, False)
    train_steps = 0
    writer = SummaryWriter(training_folder)
    train_iter = dataloader.train_iter(batch_size=FLAGS.taco_batch_size)
    nb_frames = 0
    curr_best = np.inf
    train_stats = DictList()

    # test dataloader
    test_envs = set(FLAGS.test_envs) - set(FLAGS.envs)
    test_dataloader = None if len(test_envs) == 0 else Dataloader(test_envs, FLAGS.il_val_ratio)
    scheduler = DropoutScheduler()
    while True:
        if train_steps > FLAGS.taco_train_steps:
            logging.info('Reaching maximum steps')
            break

        if train_steps % FLAGS.taco_eval_freq == 0:
            val_metrics = evaluate_loop(dataloader, model, dropout_p=scheduler.dropout_p)
            logging_metrics(nb_frames, train_steps, val_metrics, writer, 'val')

            if test_dataloader is not None:
                test_metrics = evaluate_loop(test_dataloader, model, dropout_p=scheduler.dropout_p)
                logging_metrics(nb_frames, train_steps, test_metrics, writer, 'test')

            avg_loss = [val_metrics[env_name].loss for env_name in val_metrics]
            avg_loss = np.mean(avg_loss)
            if avg_loss < curr_best:
                curr_best = avg_loss
                logging.info('Save Best with loss: {}'.format(avg_loss))
                # Save the checkpoint
                with open(os.path.join(training_folder, 'bot_best.pkl'), 'wb') as f:
                    torch.save(model, f)

        model.train()
        train_batch, train_lengths = train_iter.__next__()
        if FLAGS.cuda:
            train_batch.apply(lambda _t: _t.cuda())
            train_lengths = train_lengths.cuda()
        start = time.time()
        train_outputs = teacherforce_batch(modular_p=model,
                                           trajs=train_batch,
                                           lengths=train_lengths,
                                           decode=False,
                                           dropout_p=scheduler.dropout_p)
        optimizer.zero_grad()
        train_outputs['loss'].backward()
        optimizer.step()
        train_steps += 1
        scheduler.step()
        nb_frames += train_lengths.sum().item()
        end = time.time()
        fps = train_lengths.sum().item() / (end - start)
        train_outputs['fps'] = torch.tensor(fps)

        train_outputs = DictList(train_outputs)
        train_outputs.apply(lambda _t: _t.item())
        train_stats.append(train_outputs)

        if train_steps % FLAGS.taco_eval_freq == 0:
            train_stats.apply(lambda _tensors: np.mean(_tensors))
            logger_str = ['[TRAIN] steps={}'.format(train_steps)]
            for k, v in train_stats.items():
                logger_str.append("{}: {:.4f}".format(k, v))
                writer.add_scalar('train/' + k, v, global_step=nb_frames)
            logging.info('\t'.join(logger_str))
            train_stats = DictList()
            writer.flush()


def logging_metrics(nb_frames, train_steps, val_metrics, writer, prefix):
    for env_name, metric in val_metrics.items():
        line = ['[{}][{}] steps={}'.format(prefix, env_name, train_steps)]
        for k, v in metric.items():
            line.append('{}: {:.4f}'.format(k, v))
        logging.info('\t'.join(line))
    mean_val_metric = DictList()
    for metric in val_metrics.values():
        mean_val_metric.append(metric)
    mean_val_metric.apply(lambda t: torch.mean(torch.tensor(t)))
    for k, v in mean_val_metric.items():
        writer.add_scalar(prefix + '/' + k, v.item(), nb_frames)
    writer.flush()


