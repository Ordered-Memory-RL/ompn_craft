"""
Global settings
"""
import os
import git

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(root_dir, 'dataset')
EXP_DIR = os.path.join(root_dir, 'experiment')

repo = git.Repo(path=root_dir)
sha = repo.head.object.hexsha
CURR_VERSION = sha[:10]


__all__ = [n for n in globals() if n.isupper()]