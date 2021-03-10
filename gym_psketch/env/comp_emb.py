import os
import yaml
import numpy as np

__all__ = ['ENV_EMB', 'NB_SUBTASKS', 'ID2SKETCHS', 'ID2SKETCHLEN']

with open(os.path.join(os.path.dirname(__file__), 'envs.yaml')) as f:
    env_config = yaml.safe_load(f)

nb_env = len(env_config)
nb_tasks = max([len(env_config[name]['sketchs']) for name in env_config])
ENV_EMB = np.zeros([nb_env, nb_tasks], dtype=np.int32)
all_task_names = set()
for env_id, name in enumerate(env_config):
    for sub_task_id, sub_task in enumerate(env_config[name]['sketchs']):
        task_name = sub_task.split()[-1]
        all_task_names.add(task_name)
all_task_names = list(all_task_names)
NB_SUBTASKS = len(all_task_names)
ID2SKETCHS = []
for env_id, name in enumerate(env_config):
    subtasks = []
    for sub_task_id, sub_task in enumerate(env_config[name]['sketchs']):
        task_name = sub_task.split()[-1]
        task_id = all_task_names.index(task_name) + 1
        ENV_EMB[env_id, sub_task_id] = task_id
        subtasks.append(task_id - 1)
    ID2SKETCHS.append(subtasks)
ENV_EMB = ENV_EMB.reshape([nb_env, -1])
ID2SKETCHLEN = [len(sketchs) for sketchs in ID2SKETCHS]
