from .craft import CraftWorld
from .register import register
import sys
import inspect
import os
import yaml

global_dict = globals()


def env_factory(class_name, goal, attributes):
    def __init__(self, width=10, height=10, dense_reward=True):
        CraftWorld.__init__(self, goal, width=width, height=height,
                            dense_reward=dense_reward, fullobs=False)
    attributes['__init__'] = __init__
    return type(class_name, (CraftWorld,), attributes)


def full_env_factory(class_name, goal, attributes):
    def __init__(self, width=10, height=10, dense_reward=True):
        CraftWorld.__init__(self, goal, width=width, height=height,
                            dense_reward=dense_reward, fullobs=True)
    attributes['__init__'] = __init__
    return type(class_name, (CraftWorld,), attributes)


with open(os.path.join(os.path.dirname(__file__), 'envs.yaml')) as f:
    env_config = yaml.safe_load(f)
    for env_id, name in enumerate(env_config):
        class_name = '_'.join(name.split())
        env_cls = env_factory(class_name=class_name,
                              goal=env_config[name]['goal'],
                              attributes={'instructions': name,
                                          'sketchs': env_config[name]['sketchs'],
                                          'env_id': env_id,
                                          'cached_tiles': {}})
        global_dict[class_name] = env_cls
        gym_name = ''.join(name.split()) + '-v0'
        register(id=gym_name,
                 entry_point='gym_psketch.env.envs:' + class_name)

        class_name = class_name + '_full'
        env_cls = full_env_factory(class_name=class_name,
                                   goal=env_config[name]['goal'],
                                   attributes={'instructions': name,
                                               'sketchs': env_config[name]['sketchs'],
                                               'env_id': env_id,
                                               'cached_tiles': {}})
        global_dict[class_name] = env_cls
        gym_name = ''.join(name.split()) + 'full' + '-v0'
        register(id=gym_name,
                 entry_point='gym_psketch.env.envs:' + class_name)
