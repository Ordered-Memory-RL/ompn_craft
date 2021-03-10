import random

import numpy as np
import gym_psketch
from gym_psketch.utils import DictList
from gym_psketch.env.craft import CraftWorld, CraftState, neighbors, Actions, ID2DIR
from gym_psketch.env.cookbook import COOKBOOK as CB

__all__ = ['DemoBot']


class DemoBot:
    def __init__(self, env: CraftWorld):
        self.width = env.width
        self.height = env.height
        self.sketch = env.sketchs

        self.stack = None
        self.prev_action = None
        self.reset()

    def reset(self):
        self.stack = [GoToSubgoal(width=self.width, height=self.height,
                                  target=s.split()[-1]) for s in self.sketch]
        self.prev_action = None

    def get_action(self, obs):
        obs = DictList(obs)
        # Update subgoals
        for goal in self.stack:
            goal.update(obs)

        # Keep popping until unsatisfied subgoal
        while len(self.stack) > 0:
            curr_goal = self.stack[0]
            if not curr_goal.satisfy(self.prev_action):
                break
            self.stack.pop(0)

        # If stack is empty, emit done
        if len(self.stack) == 0:
            return Actions.DONE.value

        while True:
            curr_goal = self.stack[0]
            action = curr_goal.act_or_plan(obs)

            # If find good action, return
            if isinstance(action, Subgoal):
                self.stack.insert(0, action)
            else:
                break
        self.prev_action = action
        return action


class Subgoal:
    def __init__(self, width, height, target, state=None):
        self.width = width
        self.height = height
        self.target = target
        self.target_id = CB.object2id(target)
        if state is None:
            grid = np.ones([width, height]) * -1
            self.state = CraftState(grid=grid)
        else:
            self.state = state

    def satisfy(self, prev_action):
        """ Return a bool """
        raise NotImplementedError

    def update(self, obs: DictList):
        """ Update inner state from observation """
        # Update the grid
        img_w, img_h = obs.img.shape[0], obs.img.shape[1]
        pos = obs.pos
        dir_id = obs.dir_id

        # If fully observable
        if img_w == self.width and img_h == self.height:
            self.state.grid = obs.img.copy()

        # partial grid
        else:
            partial_grid = obs.img.copy()
            window_w, window_h = partial_grid.shape[0], partial_grid.shape[1]
            hw, hh = partial_grid.shape[0] // 2, partial_grid.shape[1] // 2

            init_left = pos[0] - hw
            left = max(init_left, 0)
            sleft = left - init_left

            init_up = pos[1] - hh
            up = max(init_up, 0)
            sup = up - init_up

            init_right = pos[0] + hw + 1
            right = min(init_right, self.width)
            sright = window_w - (init_right - right)

            init_down = pos[1] + hh + 1
            down = min(init_down, self.height)
            sdown = window_h - (init_down - down)
            self.state.grid[left:right, up:down] = partial_grid[sleft:sright, sup:sdown]

        self.state.dir = ID2DIR[dir_id]
        self.state.inventory = obs.inventory
        self.state.pos = obs.pos

    def act_or_plan(self, obs: DictList):
        raise NotImplementedError


class Explore(Subgoal):
    def satisfy(self, prev_action):
        # If target is in the current grid
        nb_found = np.array(((self.state.grid == self.target_id).nonzero())).shape[1]
        if nb_found >= 1:
            return True
        else:
            return False

    def act_or_plan(self, obs: DictList):
        unknown_coord = np.array((self.state.grid == -1).nonzero()).transpose()
        if len(unknown_coord) == 0:
            raise PlanningError("Target is unreachable")
        else:
            unknown_coord = unknown_coord[0]
        dirs = ['right', 'left', 'up', 'down']
        dir_arrs = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)])
        all_actions = [Actions.RIGHT.value,
                       Actions.LEFT.value,
                       Actions.UP.value,
                       Actions.DOWN.value]
        candidate_actions = []
        for action, dir, dir_arr in zip(all_actions, dirs, dir_arrs):
            neighbor_coord = neighbors(pos=self.state.pos,
                                       width=self.width,
                                       height=self.height,
                                       dir=dir)[0]
            thing = self.state.grid[neighbor_coord]

            # If not passable, don't insert in candidate
            if thing != 0:
                continue

            hamming_dist = _hamming(self.state.pos, unknown_coord)
            new_hamming_dist = _hamming(self.state.pos + dir_arr, unknown_coord)
            if new_hamming_dist < hamming_dist:
                candidate_actions.append(action)

        if len(candidate_actions) == 0:
            candidate_actions = [Actions.UP.value,
                                 Actions.DOWN.value,
                                 Actions.LEFT.value,
                                 Actions.RIGHT.value]
        return random.choice(candidate_actions)

    def __repr__(self):
        return "Explore(target={})".format(self.target)


class GoToSubgoal(Subgoal):
    def __init__(self, width, height, target, state=None):
        super(GoToSubgoal, self).__init__(width=width, height=height,
                                          target=target, state=state)
        self.available_target_pos = None
        self.curr_plan = None
        self.prev_inventory = None
        self.water_id = CB.object2id('water')
        self.stond_id = CB.object2id('stone')

    def __repr__(self):
        return "GoTo(target={})".format(self.target)

    def plan(self, state):
        """ Randomly Pick a nearest grabbale indice """
        # Find the grabbable indices surrounding curr pos
        subgoal = Explore(width=self.width, height=self.height,
                          target=self.target, state=state)
        return subgoal

    def act_or_plan(self, obs):
        """ Return precedding subgoal or the action"""
        state = self.state
        pos, dir = state.pos, state.dir
        self.prev_inventory = state.inventory.copy()
        width, height = state.grid.shape[0], state.grid.shape[1]
        if self.curr_plan is None or not _cmp_pos(pos, self.curr_plan[0]):
            self.curr_plan = self.try_find_curr_plan(state)

        # No plan found
        if self.curr_plan is None:
            return self.plan(state)

        # If facing the target
        if _cmp_pos(neighbors(state.pos, width, height, dir)[0], self.curr_plan[-1]):
            return Actions.USE.value

        next_pos = self.curr_plan[1]
        next_thing = state.grid[next_pos]

        # If empty
        if next_thing == 0:
            action = _get_action(init_pos=pos,
                                 after_pos=self.curr_plan[1])
            self.curr_plan.pop(0)

        else:
            if next_thing == self.water_id or next_thing == self.stond_id or next_thing == self.target_id:
                # If facing that "thing", USE. If not, turn
                if _cmp_pos(neighbors(state.pos, width, height, dir)[0], next_pos):
                    action = Actions.USE.value
                else:
                    action = _get_action(init_pos=pos,
                                         after_pos=self.curr_plan[1])
            else:
                raise ValueError

        return action

    def try_find_curr_plan(self, state: CraftState):
        # If there is no available target pos, generate one
        if self.available_target_pos is None or len(self.available_target_pos) == 0:
            self.available_target_pos = _find_all(state, self.target).tolist()
            if len(self.available_target_pos) == 0:
                return None

        chosen = None
        for candidate in self.available_target_pos:
            ignore_water_or_stone = self.target == 'gold' or self.target == 'gem'
            path = _find_shortest_path(state, target_pos=candidate,
                                       ignore_water_and_stone=ignore_water_or_stone)
            if path is not None:
                if chosen is None:
                    chosen = path
                elif len(path) < len(chosen):
                    chosen = path
        return chosen

    def satisfy(self, prev_action) -> bool:
        if self.state.inventory is None:
            return False

        if self.target_id in CB.grabbable_indices:
            return self.state.inventory[self.target_id] > 0

        # Satisfiability of use
        # True if facing the workshop and prev_action is True
        else:
            if prev_action is None:
                return False

            coord = neighbors(pos=self.state.pos, dir=self.state.dir,
                              width=self.width, height=self.height)[0]
            thing = self.state.grid[coord]
            if thing == self.target_id and prev_action == Actions.USE.value:
                inventory_diff = self.state.inventory - self.prev_inventory
                return (inventory_diff != 0).sum() > 0
            else:
                return False


def _find_shortest_path_pos(things, accept_fn, init_pos):
    """
    :param things: A [x * y] map. 0 means passable,
    :param accept_fn: Test whether the
    :return: A path to one of pos that accept_fn returns True, or None
    path[0] == init_pos, accept_fn(path[1]) == True
    """
    assert things[init_pos] == 0, "Current position should be empty"
    width, height = things.shape[0], things.shape[1]
    searched = np.zeros_like(things)
    searched[init_pos] = 1
    parents = np.ones(list(things.shape) + [2], dtype=int) * -1
    queue = [init_pos]
    curr_pos = init_pos
    while len(queue) > 0:
        curr_pos = queue.pop(0)
        if accept_fn(curr_pos):
            break

        # Expandible positions if not searched and reachable
        next_poss = [next_pos for next_pos in neighbors(curr_pos, width, height)
                     if searched[next_pos] == 0]

        # For each next pos, mark the parents and searched
        for next_pos in next_poss:
            if accept_fn(next_pos):
                parents[next_pos] = curr_pos
                searched[next_pos] = 1
                queue.insert(-1, next_pos)
            elif things[next_pos] == 0:
                parents[next_pos] = curr_pos
                searched[next_pos] = 1
                queue.insert(-1, next_pos)

    # Not reachable
    if not accept_fn(curr_pos):
        return None

    path = [curr_pos]
    while not _cmp_pos(path[0], init_pos):
        parent = tuple(parents[path[0]].tolist())
        path.insert(0, parent)

    for coord in path[:-1]:
        assert things[coord] == 0
    assert _cmp_pos(path[0], init_pos) and accept_fn(path[-1])
    return path


def _find_shortest_path(state: CraftState, target_pos, ignore_water_and_stone=False):
    """ Return a sequence of postions that reach target_pos or None if not reachable """
    pos, dir, grid = state.pos, state.dir, state.grid

    def _accept(_pos):
        return _cmp_pos(_pos, target_pos)

    assert grid[pos] == 0, "Current position should be empty"

    # make 'water' and 'stone' passable, since we can build a bridge.
    # Otherwise treat it as unpassable
    if ignore_water_and_stone:
        water_coords = _find_all(state, 'water')
        stone_coords = _find_all(state, 'stone')
        for water_coord in water_coords:
            grid[tuple(water_coord.tolist())] = 0
        for stone_coord in stone_coords:
            grid[tuple(stone_coord.tolist())] = 0

    path = _find_shortest_path_pos(things=grid, accept_fn=_accept, init_pos=state.pos)
    return path


def _get_action(init_pos, after_pos):
    dx = after_pos[0] - init_pos[0]
    dy = after_pos[1] - init_pos[1]
    if dx == 0 and dy == -1:
        return Actions.DOWN.value
    elif dx == 0 and dy == 1:
        return Actions.UP.value
    elif dx == -1 and dy == 0:
        return Actions.LEFT.value
    elif dx == 1 and dy == 0:
        return Actions.RIGHT.value
    else:
        raise ValueError


def _hamming(pos1, pos2):
    return np.abs(pos1[0] - pos2[0]) + np.abs(pos1[1] - pos2[1])


def _cmp_pos(pos1, pos2):
    return pos1[0] == pos2[0] and pos1[1] == pos2[1]


def _find_all(state: CraftState, target):
    target_id = CB.object2id(target)
    return np.array(np.where(state.grid == target_id)).transpose()


def _pretty_print(np_grid):
    lines = []
    width, height = np_grid.shape[0], np.shape[1]
    for y in reversed(range(height)):
        line = []
        for x in range(width):
            line.append(str(np_grid[x, y]))
        lines.append('\n'.join(line))
    return '\n'.join(lines)


class PlanningError(Exception):
    pass


def generate_one_traj(bot: DemoBot, env, render_mode="ansi"):
    use_viz = render_mode is not None
    traj = gym_psketch.DictList({})
    obs = env.reset()
    bot.reset()
    done = False
    ret = 0
    viz = []
    while not done:
        action = bot.get_action(obs)
        if use_viz:
            viz.append(env.render(render_mode))
        next_obs, reward, done, _ = env.step(action)
        ret += reward
        transition = {'action': gym_psketch.ID2ACTIONS[action],
                      'reward': reward}
        transition['features'] = obs['features']
        transition = gym_psketch.DictList(transition)
        traj.append(transition)
        obs = next_obs

    # Append Done Action
    transition = {'action': gym_psketch.Actions.DONE,
                  'reward': 0}
    if use_viz:
        viz.append(env.render(render_mode))
    transition['features'] = obs['features']
    transition = gym_psketch.DictList(transition)
    traj.append(transition)
    if not use_viz:
        return ret, {k: v for k,v in traj.items()}
    else:
        return ret, {k: v for k,v in traj.items()}, viz
