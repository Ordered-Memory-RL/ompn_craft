import numpy as np

from gym_psketch import ACTION_VOCAB

CHARS = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]


def visual(val, max_val):
    val = np.clip(val, 0, max_val)
    if abs(val) == max_val:
        step = len(CHARS) - 1
    else:
        step = int(abs(float(val) / max_val) * len(CHARS))
    colourstart = ""
    colourend = ""
    if val < 0:
        colourstart, colourend = '\033[90m', '\033[0m'
    return colourstart + CHARS[step] + colourend


def idxpos2tree(idxs, pos):
    viz = pos.transpose(1, 0).cpu().numpy()
    string = np.array2string(viz, formatter={'float_kind': lambda x: visual(x, 1)},
                             max_line_width=5000)
    lines = [line[2:-1].replace(']', '') for line in string.split('\n')]
    lines.append(' '.join([ACTION_VOCAB[idx] for idx in idxs]))
    return '\n'.join(lines)


def distance2ctree(depth, sen, binary=False):
    assert len(depth) == len(sen) - 1
    if len(sen) == 1:
        parse_tree = sen[0]
    else:
        max_depth = max(depth)
        parse_tree = []
        sub_depth = []
        sub_sen = []
        for d, w in zip(depth, sen[:-1]):
            sub_sen.append(w)
            if d == max_depth:
                parse_tree.append(distance2ctree(sub_depth, sub_sen, binary))
                sub_depth = []
                sub_sen = []
            else:
                sub_depth.append(d)
        sub_sen.append(sen[-1])
        parse_tree.append(distance2ctree(sub_depth, sub_sen, binary))
    if len(parse_tree) > 2 and binary:
        bin_tree = parse_tree.pop(-1)
        while len(parse_tree) > 0:
            bin_tree = [parse_tree.pop(-1), bin_tree]
        return bin_tree
    return parse_tree


def tree_to_str(parse_tree):
    if isinstance(parse_tree, list):
        res = [tree_to_str(ele) for ele in parse_tree]
        return '(' + ' '.join(res) + ')'
    elif isinstance(parse_tree, str):
        return parse_tree
