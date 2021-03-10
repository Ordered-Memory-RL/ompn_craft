from . import array
from .cookbook import COOKBOOK as CB
from .rendering import *
from .window import Window
import enum
import numpy as np
from skimage.measure import block_reduce
from gym import Env
import attr
import matplotlib.pyplot as plt

__all__ = ['Actions', 'ID2ACTIONS', 'CraftWorld', 'CraftState', 'ACTION_VOCAB']


class Actions(enum.Enum):
    DOWN = 0
    UP = 1
    LEFT = 2
    RIGHT = 3
    USE = 4
    DONE = 5

ID2DIR = ['down', 'up', 'left', 'right']
ID2ACTIONS = [a for a in Actions]
ACTION_VOCAB = ['↓', '↑', '←', '→', 'u', 'D']


def random_free(grid, random, width, height):
    pos = None
    while pos is None:
        (x, y) = (random.randint(width), random.randint(height))
        if grid[x, y] != 0:
            continue
        pos = (x, y)
    return pos


def neighbors(pos, width, height, dir=None):
    x, y = pos
    res = []
    if x > 0 and (dir is None or dir == 'left'):
        res.append((x-1, y))
    if y > 0 and (dir is None or dir == 'down'):
        res.append((x, y-1))
    if x < width - 1 and (dir is None or dir == 'right'):
        res.append((x+1, y))
    if y < height - 1 and (dir is None or dir == 'up'):
        res.append((x, y+1))
    return res


@attr.s
class CraftState:
    """ Internal state for each episode """
    inventory = attr.ib(None)
    pos = attr.ib(None)
    dir = attr.ib(None)
    grid = attr.ib(None)
    cache = attr.ib(None)
    sketch_id = attr.ib(None)


class CraftWorld(Env):
    metadata = {'render.modes': ['ansi']}

    # Some meta data for each env
    instruction = None
    sketchs = None
    env_id = None
    cached_tiles = {}

    def __init__(self, goal, width=10, height=10, window_width=5,
                 window_height=5, num_ing=1, dense_reward=True, fullobs=False):
        assert goal in CB.possible_goals, "Invalid Goals"
        self.fullobs = fullobs
        self.num_ing = num_ing
        self.dense_reward = dense_reward
        self.width = width
        self.height = height
        self.window_width = window_width
        self.window_height = window_height
        self.n_features = 2 * window_width * window_height * CB.n_kinds + \
                          CB.n_kinds + \
                          4
        self.n_actions = Actions.__len__()

        self.non_grabbable_indices = CB.non_grabbable_indices
        self.grabbable_indices = CB.grabbable_indices
        self.workshop_indices = CB.workshops
        self.water_index = CB.index["water"]
        self.stone_index = CB.index["stone"]
        self.random = np.random.RandomState(None)

        # Set goal ids
        self.goal_id = CB.index[goal]
        self.make_island = self.goal_id == CB.index['gold']
        self.make_cave = self.goal_id == CB.index['gem']

        # State variable for each episode
        self.state = None

    def seed(self, seed=None):
        np.random.seed(seed)
        self.random = np.random.RandomState(seed)

    def reset(self):
        grid = self._gen_grid(make_island=self.make_island,
                              make_cave=self.make_cave)
        self.state = CraftState(inventory=np.zeros(CB.n_kinds),
                                pos=random_free(grid=grid,
                                                random=self.random,
                                                width=self.width,
                                                height=self.height),
                                dir=self.random.choice(ID2DIR),
                                grid=grid,
                                cache=None,
                                sketch_id=0)
        return self._obs()

    def step(self, action):
        if isinstance(action, int):
            action = ID2ACTIONS[action]
        prev_inventory = self.state.inventory.copy()
        self.state = self._update_grid_and_inventory(self.state, action)
        reward = self._reward(action, prev_inventory) if self.dense_reward else 0
        done = self.satisfy()
        return self._obs(), reward, done, {}

    def _reward(self, action, prev_inventory):
        if self.state.sketch_id >= len(self.sketchs):
            return 0
        sketch = self.sketchs[self.state.sketch_id]
        target = sketch.split()[-1]
        target_id = CB.object2id(target)
        if action == Actions.USE:
            inventory_diff = self.state.inventory - prev_inventory
            if target_id in self.grabbable_indices:
                satisfy = inventory_diff[target_id] > 0
            elif target_id in self.workshop_indices:
                front_coord = neighbors(self.state.pos, self.width, self.height, self.state.dir)[0]
                front_thing = self.state.grid[front_coord]
                correct_workshop = front_thing == target_id
                use_success = (inventory_diff != 0).sum() > 0
                satisfy = correct_workshop and use_success
            else:
                raise ValueError('Invalid target id', target_id)
        else:
            satisfy = False

        reward = 1 if satisfy else 0
        if satisfy:
            self.state.sketch_id += 1
        return reward

    def satisfy(self):
        return (self.state.inventory[self.goal_id] > 0).item()

    def _update_grid_and_inventory(self, state: CraftState, action: Actions) -> CraftState:
        """ Update agent state """
        x, y = state.pos
        n_dir = state.dir
        n_inventory = state.inventory
        n_grid = self.state.grid

        # move actions
        if action == Actions.DOWN:
            dx, dy = (0, -1)
            n_dir = 'down'
        elif action == Actions.UP:
            dx, dy = (0, 1)
            n_dir = 'up'
        elif action == Actions.LEFT:
            dx, dy = (-1, 0)
            n_dir = 'left'
        elif action == Actions.RIGHT:
            dx, dy = (1, 0)
            n_dir = 'right'
        elif action == Actions.DONE:
            dx, dy = (0, 0)

        # use actions
        elif action == Actions.USE:
            dx, dy = (0, 0)
            nx, ny = neighbors(state.pos, dir=state.dir,
                               width=self.width,
                               height=self.height)[0]
            thing = self.state.grid[nx, ny]
            if thing != 0:
                # Copy
                n_inventory = self.state.inventory.copy()
                n_grid = self.state.grid.copy()
                if thing in self.grabbable_indices:
                    n_inventory[thing] += 1
                    n_grid[nx, ny] = 0

                elif thing in self.workshop_indices:
                    workshop = CB.index.get(thing)
                    for output, inputs in CB.recipes.items():
                        if inputs["_at"] != workshop:
                            continue
                        yld = inputs["_yield"] if "_yield" in inputs else 1
                        ing = [i for i in inputs if isinstance(i, int)]
                        if any(n_inventory[i] < inputs[i] for i in ing):
                            continue
                        n_inventory[output] += yld
                        for i in ing:
                            n_inventory[i] -= inputs[i]

                elif thing == self.water_index:
                    if n_inventory[CB.index["bridge"]] > 0:
                        n_grid[nx, ny] = 0

                elif thing == self.stone_index:
                    if n_inventory[CB.index["axe"]] > 0:
                        n_grid[nx, ny] = 0

        # other
        else:
            raise Exception("Unexpected action: %s" % action)

        n_x = x + dx
        n_y = y + dy
        if self.state.grid[n_x, n_y] != 0:
            n_x, n_y = x, y

        new_state = CraftState(pos=(n_x, n_y),
                               dir=n_dir,
                               inventory=n_inventory,
                               grid=n_grid,
                               cache=None,
                               sketch_id=state.sketch_id)
        return new_state

    def _gen_grid(self, make_island=False, make_cave=False):
        # generate grid
        grid = np.zeros((self.width, self.height), dtype=int)
        i_bd = CB.index["boundary"]
        grid[0, :] = i_bd
        grid[self.width - 1:, :] = i_bd
        grid[:, 0] = i_bd
        grid[:, self.height - 1:] = i_bd

        # treasure
        if make_island or make_cave:
            (gx, gy) = (1 + self.random.randint(self.width - 2), 1)
            treasure_index = CB.index["gold"] if make_island else CB.index["gem"]
            wall_index = self.water_index if make_island else self.stone_index
            grid[gx, gy] = treasure_index
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if grid[gx + i, gy + j] == 0:
                        grid[gx + i, gy + j] = wall_index

        # ingredients
        for primitive in CB.primitives:
            if primitive == CB.index["gold"] or \
                    primitive == CB.index["gem"]:
                continue
            for i in range(self.num_ing):
                (x, y) = random_free(grid, self.random, self.width, self.height)
                grid[x, y] = primitive

        # generate crafting stations
        for ws_id in self.workshop_indices:
            ws_x, ws_y = random_free(grid, self.random, self.width, self.height)
            grid[ws_x, ws_y] = ws_id

        # generate init pos
        return grid

    def _obs(self):
        if self.state is None:
            raise ValueError

        if self.state.cache is None:
            x, y = self.state.pos
            hw = self.window_width // 2
            hh = self.window_height // 2
            bhw = (self.window_width * self.window_width) // 2
            bhh = (self.window_height * self.window_height) // 2

            oh_grid = np.eye(CB.n_kinds)[self.state.grid.reshape(-1)].reshape([self.width, self.height, -1])
            grid_feats = array.pad_slice(oh_grid, (x - hw, x + hw + 1),
                                         (y-hh, y+hh+1), pad_value=0)
            if self.fullobs:
                grid_feats_big = array.pad_slice(oh_grid, (x - bhw, x + bhw + 1),
                                                 (y-bhh, y+bhh+1), pad_value=0)
                grid_feats_big_red = block_reduce(grid_feats_big,
                        (self.window_width, self.window_height, 1), func=np.max)
            else:
                grid_feats_big_red = np.zeros_like(grid_feats)
            #pos_feats = np.asarray(self.state.pos)
            #pos_feats[0] /= self.width
            #pos_feats[1] /= self.height
            dir_features = np.zeros(4)
            dir_features[ID2DIR.index(self.state.dir)] = 1
            features = np.concatenate((grid_feats.ravel(),
                                       grid_feats_big_red.ravel(), self.state.inventory,
                                       dir_features))
            self.state.cache = {'features': features, 'inventory': self.state.inventory,
                                'pos': self.state.pos, 'dir_id': ID2DIR.index(self.state.dir),
                                'img': self.state.grid if self.fullobs
                                else array.pad_slice(self.state.grid, (x - hw, x + hw + 1),
                                                     (y-hh, y+hh+1), pad_value=0)}
        return self.state.cache
    
    def render(self, mode='ansi'):
        if mode == 'ansi':
            return self.pretty()
        elif mode == 'rgb':
            return self.get_rgb(tile_size=TILE_PIXELS)
        else:
            return super(CraftWorld, self).render(mode=mode)

    def get_rgb(self, tile_size):
        window = Window('test')
        assert self.state is not None
        grid = self.state.grid

        width_px = self.width * tile_size
        height_px = self.height * tile_size
        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        if not self.fullobs:
            pos_x, pos_y = self.state.pos
            hw = self.window_width // 2
            hh = self.window_height // 2
            x_range = [i for i in range(pos_x - hw, pos_x + hw + 1)]
            y_range = [i for i in range(pos_y - hh, pos_y + hh + 1)]
        else:
            x_range = [i for i in range(self.width)]
            y_range = [i for i in range(self.height)]

        for y in reversed(range(self.height)):
            for x in range(self.width):
                if x in x_range and y in y_range:
                    cell = grid[x, y]
                    tile_img = self.render_cell(cell,
                                                tile_size=tile_size,
                                                has_agent=(x,y) == self.state.pos,
                                                direction=self.state.dir)

                    ymin = y * tile_size
                    ymax = (y+1) * tile_size
                    xmin = x * tile_size
                    xmax = (x+1) * tile_size
                    img[ymin:ymax, xmin:xmax, :] = tile_img

        window.show_img(img)

        # Get Inventory String
        inventory_str = ["Inventory:"]
        for inventory_id, val in enumerate(self.state.inventory):
            if val > 0:
                inventory_str.append("{}: {}".format(CB.index.get(inventory_id),
                                                     int(val)))

        window.set_caption('  '.join(inventory_str))
        fig = window.fig
        w, h = fig.canvas.get_width_height()
        fig.tight_layout()
        fig.canvas.draw()
        p_img = np.fromstring(fig.canvas.tostring_rgb(),
                              dtype=np.uint8).reshape(h, w, 3)
        window.close()
        return p_img

    @classmethod
    def render_cell(cls, cell, tile_size, subdivs=1, has_agent=False, direction=0):
        key = cls._tile_key(cell, has_agent, dir=direction)
        if key in cls.cached_tiles:
            return cls.cached_tiles[key]

        # draw wall
        try:
            if CB.id2object(cell) == 'boundary':
                img = np.ones(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8) * 0
            else:
                img = load_icons(CB.id2object(cell), tile_size * subdivs, tile_size * subdivs)
        except FileNotFoundError:
            img = np.ones(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8) * 255

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        # Plot agent
        if has_agent:
            tri_fn = point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
            )

            # Rotate the agent based on its direction
            dir_id = RENDER_DIRID[direction]
            tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * dir_id)
            fill_coords(img, tri_fn, COLORS['red'])

        img = downsample(img, subdivs)
        cls.cached_tiles[key] = img
        return img

    @staticmethod
    def _tile_key(cell, has_agent, dir):
        if has_agent:
            return '{}_{}'.format(CB.id2object(cell), dir)
        else:
            return CB.id2object(cell)

    def pretty(self):
        """ Pretty print to strings """
        if self.state is None:
            return ""

        # Grid to string
        lines = []
        if not self.fullobs:
            pos_x, pos_y = self.state.pos
            hw = self.window_width // 2
            hh = self.window_height // 2
            x_range = [i for i in range(pos_x - hw, pos_x + hw + 1)]
            y_range = [i for i in range(pos_y - hh, pos_y + hh + 1)]
        else:
            x_range = [i for i in range(self.width)]
            y_range = [i for i in range(self.height)]
        for y in reversed(range(self.height)):
            line = []
            for x in range(self.width):
                # Empty if out of boundary
                if x in x_range and y in y_range:
                    # Plot agent
                    if (x, y) == self.state.pos:
                        if self.state.dir == 'left':
                            line.append("<@")
                        elif self.state.dir == 'right':
                            line.append("@>")
                        elif self.state.dir == 'up':
                            line.append("^@")
                        elif self.state.dir == 'down':
                            line.append("@v")
                        else:
                            raise ValueError

                    # Plot that thing
                    else:
                        cell = self.state.grid[x, y]
                        if cell == 0:
                            line.append("  ")
                        else:
                            obj_str = CB.index.get(cell)
                            line.append(obj_str[0] + obj_str[-1])
                else:
                    line.append(" ")

            lines.append(' '.join(line))

        # Plot Inventory
        lines.append("")
        lines.append('Inventory:')
        for inventory_id, val in enumerate(self.state.inventory):
            if val > 0:
                lines.append("{}: {}".format(CB.index.get(inventory_id),
                                             val))
        return '\n'.join(lines)
