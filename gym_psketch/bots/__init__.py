from .demo_bot import DemoBot
from .model_bot import *
from .omdec import *


def make(arch, vec_size, action_size, hidden_size, nb_slots, env_arch):
    if arch == 'mlp':
        bot = MLPBot(vec_size=vec_size, action_size=action_size,
                     hidden_size=hidden_size, env_arch=env_arch)

    elif arch == 'lstm':
        bot = LSTMBot(vec_size=vec_size, action_size=action_size,
                      hidden_size=hidden_size, num_layers=nb_slots,
                      env_arch=env_arch)
    elif arch == 'omstack':
        bot = OMStackBot(vec_size=vec_size, action_size=action_size,
                         slot_size=hidden_size, nb_slots=nb_slots,
                         env_arch=env_arch)
    else:
        raise ValueError('No such architecture')
    return bot


