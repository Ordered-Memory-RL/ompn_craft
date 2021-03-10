from .env import *
from .bots import *
from .settings import *
from .dataloader import *
from .evaluate import *
from .utils import *


def load(config):
    cls_name = config.world.name
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception("No such world: {}".format(cls_name))
