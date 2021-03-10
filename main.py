"""
Main entry
"""
import sys
from absl import flags, logging
from absl import app
import time
import datetime
import os
import traceback
import string
import torch
import random

import gym_psketch.bots.model_bot
import imitation as IL
import generate_demo as demo
from taco import train_taco
from compile import train_compile

FLAGS = flags.FLAGS
flags.DEFINE_list('envs', default=['makebedfull-v0'], help='List of env to train. Use comma spearated')
flags.DEFINE_list('test_envs', default=[], help='Extra test envs')
flags.DEFINE_integer('max_steps', default=64, help='maximum environment steps')
flags.DEFINE_integer('width', default=10, help='width of env')
flags.DEFINE_integer('height', default=10, help='height of env')
flags.DEFINE_enum('arch', enum_values=['mlp', 'lstm', 'omstack'],
                  default='lstm', help='Architecture')
flags.DEFINE_integer('eval_episodes', default=30, help='Evaluation episode number')

# Model
flags.DEFINE_integer('hidden_size', default=64, help='model hidden size')
flags.DEFINE_integer('nb_slots', default=1, help='model hidden size')
flags.DEFINE_enum('env_arch', enum_values=['emb', 'sketch', 'noenv', 'grusketch'],
                  default='emb', help='env encoder Architecture')

# Misc
flags.DEFINE_bool('debug', default=False, help='Flag for debug mode')
flags.DEFINE_enum('mode', default='IL', enum_values=['IL', 'demo', 'compile',
                                                     'taco'],
                  help='choosing between IL and baselines')
flags.DEFINE_string('experiment', default=None, help='Name of experiment')
flags.DEFINE_bool('cuda', default=False, help='Use cuda')
flags.DEFINE_integer('procs', default=4, help='Number of process')


def handler(type, value, tb):
    logging.exception("Uncaught exception: %s", str(value))
    logging.exception("\n".join(traceback.format_exception(type, value, tb)))


def random_string():
    return ''.join(random.sample(string.ascii_lowercase + string.ascii_uppercase, k=10))


def setup_logging_and_exp_folder():
    # Random string if debug
    if FLAGS.debug:
        FLAGS.experiment = "{}_{}".format(FLAGS.mode, random_string())

    # Use time stamp or user specified if not debug
    else:
        ts = time.time()
        FLAGS.experiment = FLAGS.experiment if FLAGS.experiment is not None else \
            "{}_{}".format(FLAGS.mode,
                           datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M'))
    training_folder = os.path.join(gym_psketch.EXP_DIR, FLAGS.experiment)

    # Create train folder
    if os.path.exists(training_folder):
        print('{} exists!'.format(training_folder))
        exit(-1)
    else:
        os.makedirs(training_folder, exist_ok=False)

    # set up logging
    if FLAGS.debug:
        logging.get_absl_handler().python_handler.stream = sys.stdout
    else:
        logging.get_absl_handler().use_absl_log_file('absl_logging', training_folder)
    return training_folder


def main(_):
    trainig_folder = setup_logging_and_exp_folder()
    FLAGS.cuda = FLAGS.cuda and torch.cuda.is_available()
    logging.info('Use Cuda: {}'.format(FLAGS.cuda))
    logging.info('Current git SHA: ' + gym_psketch.CURR_VERSION)

    # save options
    fpath = os.path.join(trainig_folder, 'flagfile')
    with open(fpath, 'w') as f:
        f.write(FLAGS.flags_into_string())

    if FLAGS.mode == 'IL':
        IL.run(training_folder=trainig_folder)
    elif FLAGS.mode == 'demo':
        demo.main()
    elif FLAGS.mode == 'compile':
        train_compile.main(training_folder=trainig_folder)
    elif FLAGS.mode == 'taco':
        train_taco.main(training_folder=trainig_folder)
    else:
        logging.fatal('Improper Mode {}'.format(FLAGS.mode))
    logging.info('Done')


if __name__ == '__main__':
    FLAGS(sys.argv)
    app.run(main)
