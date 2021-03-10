"""
Static Global Stuff
"""
import yaml
import os
import numpy as np


class Index:
    def __init__(self):
        self.contents = {'empty': 0}
        self.ordered_contents = ['empty']
        self.reverse_contents = {0: 'empty'}

    def __getitem__(self, item):
        return self.contents[item]

    def index(self, item):
        if item not in self.contents:
            idx = len(self.contents)
            self.ordered_contents.append(item)
            self.contents[item] = idx
            self.reverse_contents[idx] = item
        idx = self[item]
        return idx

    def get(self, idx):
        return self.reverse_contents[idx]

    def __len__(self):
        return len(self.contents)

    def __iter__(self):
        return iter(self.ordered_contents)


class Cookbook(object):

    def __init__(self, recipes_path):
        with open(recipes_path) as recipes_f:
            recipes = yaml.safe_load(recipes_f)
        # self.environment = set(recipes["environment"])
        self.index = Index()
        self.environment = set(self.index.index(e) for e in recipes["environment"])
        self.workshops = set(self.index.index(w) for w in recipes['workshops'])
        self.primitives = set(self.index.index(p) for p in recipes["primitives"])
        self.recipes = {}
        for output, inputs in recipes["recipes"].items():
            d = {}
            for inp, count in inputs.items():
                # special keys
                if "_" in inp:
                    d[inp] = count
                else:
                    d[self.index.index(inp)] = count
            self.recipes[self.index.index(output)] = d
        self.n_kinds = len(self.index)
        self.possible_goals = recipes['primitives'] + [self.index.get(o) for o in list(self.recipes.keys())]
        self.non_grabbable_indices = self.environment.union(self.workshops)
        self.grabbable_indices = [i for i in range(1, self.n_kinds)
                                  if i not in self.non_grabbable_indices]

        # Out of boundary invalid
        self.index.index('invalid')

    def primitives_for(self, goal):
        out = {}

        def insert(kind, count):
            assert kind in self.primitives
            if kind not in out:
                out[kind] = count
            else:
                out[kind] += count

        for ingredient, count in self.recipes[goal].items():
            if not isinstance(ingredient, int):
                assert ingredient[0] == "_"
                continue
            elif ingredient in self.primitives:
                insert(ingredient, count)
            else:
                sub_recipe = self.recipes[ingredient]
                n_produce = sub_recipe["_yield"] if "_yield" in sub_recipe else 1
                n_needed = int(np.ceil(1. * count / n_produce))
                expanded = self.primitives_for(ingredient)
                for k, v in expanded.items():
                    insert(k, v * n_needed)

        return out

    def id2object(self, idx):
        return self.index.get(idx)

    def object2id(self, obj):
        return self.index.index(obj)

    def info(self):
        print('environment', [self.id2object(idx) for idx in self.environment])
        print('workshop', [self.id2object(idx) for idx in self.workshops])
        print('primitive', [self.id2object(idx) for idx in self.primitives])
        print('grabbale:', [self.id2object(idx) for idx in self.grabbable_indices])
        print('nongrabbale:', [self.id2object(idx) for idx in self.non_grabbable_indices])


RECIPE_PATH = os.path.join(os.path.dirname(__file__), 'recipes.yaml')
COOKBOOK = Cookbook(RECIPE_PATH)
COOKBOOK.info()
