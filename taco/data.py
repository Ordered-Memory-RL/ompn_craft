"""
TACO dataloader
"""
from gym_psketch.settings import DATASET_DIR
from gym_psketch.utils import DictList
import os
from torch.nn.utils.rnn import pad_sequence
import pickle
from absl import logging
import random
import torch

__all__ = ['Dataloader']


class Dataloader:
    def __init__(self, env_names, val_ratio, use_bot=False):
        data = {}
        for env_name in env_names:
            if use_bot:
                pkl_name = os.path.join(DATASET_DIR, '{}_bot.pkl'.format(env_name))
            else:
                pkl_name = os.path.join(DATASET_DIR, '{}.pkl'.format(env_name))
            with open(pkl_name, 'rb') as f:
                data[env_name] = pickle.load(f)

        # Turn it into DictList
        for env_name in data:
            new_data = []
            trajs = data[env_name]['trajs']
            env_id = data[env_name]['env_id']
            for traj in trajs:
                new_traj = DictList(traj)

                # Remove done
                new_traj.apply(lambda _t: _t[:-1])
                new_traj.done = [False] * (len(new_traj) - 1) + [True]
                new_traj.action = [a.value for a in new_traj.action]
                new_traj.env_id = [env_id] * len(new_traj)
                new_data.append(new_traj)
            data[env_name] = new_data

        self.data = {'train': {}, 'val': {}}
        for env_name in env_names:
            _data = data[env_name]
            nb_data = len(_data)
            nb_val = int(val_ratio * nb_data)
            random.shuffle(_data)
            self.data['val'][env_name] = _data[:nb_val]
            self.data['train'][env_name] = _data[nb_val:]
            logging.info('{}: Train: {} Val: {}'.format(env_name, len(self.data['train'][env_name]),
                                                        len(self.data['val'][env_name])))
        self.env_names = env_names

    def train_iter(self, batch_size, env_names=None):
        all_train_trajs = []
        env_names = self.env_names if env_names is None else env_names
        for env_name in env_names:
            all_train_trajs += self.data['train'][env_name]
        return self.batch_iter(all_train_trajs, batch_size, shuffle=True, epochs=-1)

    def val_iter(self, batch_size, env_names=None):
        all_train_trajs = []
        env_names = self.env_names if env_names is None else env_names
        for env_name in env_names:
            all_train_trajs += self.data['val'][env_name]
        return self.batch_iter(all_train_trajs, batch_size, shuffle=False, epochs=1)

    def batch_iter(self, trajs, batch_size, shuffle=True, epochs=-1) -> DictList:
        """
        :param trajs: A list of DictList
        :param batch_size: int
        :param seq_len: int
        :param epochs: int. If -1, then forever
        :return: DictList [bsz, seq_len]
        """
        epoch_iter = range(1, epochs+1) if epochs > 0 else _forever()
        for _ in epoch_iter:
            if shuffle:
                random.shuffle(trajs)

            start_idx = 0
            while start_idx < len(trajs):
                batch = DictList()
                lengths = []
                for _traj in trajs[start_idx: start_idx + batch_size]:
                    lengths.append(len(_traj.action))
                    _traj.apply(lambda _t: torch.tensor(_t))
                    batch.append(_traj)

                batch.apply(lambda _t: pad_sequence(_t, batch_first=True))
                yield batch, torch.tensor(lengths)
                start_idx += batch_size


def _forever():
    i = 1
    while True:
        yield i
        i += 1
