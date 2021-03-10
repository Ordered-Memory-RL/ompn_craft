"""
This script takes a tf board folder structure like
exp
|__exp1
|    |__ run1
|    |__ run2
|    |__ run1
|    |__ run1
|    |__ run3
|    |__ ...
|    |__ runN
|
|__exp2
     |__ run1
     |__ run2
     |__ run3
     |__ ...
     |__ runN

And generate plots with shaded area for all
"""
import argparse
import os
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math


def get_empty_series(steps):
    result = Series()
    for step in steps:
        result.add(step, 0)
    return result


class Series:
    def __init__(self):
        self.values = []
        self.steps = []

    def add(self, step, val):
        """ Insert step and value. Maintain sorted w.r.t. steps """
        if len(self.steps) == 0:
            self.steps.append(step)
            self.values.append(val)
        else:
            for idx in reversed(range(len(self.steps))):
                if step > self.steps[idx]:
                    break
            else:
                idx = -1
            self.steps.insert(idx + 1, step)
            self.values.insert(idx + 1, val)

    def verify(self):
        for i in range(len(self.steps) - 1):
            assert self.steps[i] <= self.steps[i + 1]


def parse_tb_event_files(event_dir, tags):
    data = {}
    event_files = [os.path.join(event_dir, fname) for fname in os.listdir(event_dir)
                   if 'events.' in fname and not os.path.isdir(fname)]
    print('Found {} event file'.format(len(event_files)))
    for event_file in event_files:
        for e in tf.compat.v1.train.summary_iterator(event_file):
            for v in e.summary.value:
                tag = v.tag.replace('/', '_')
                if tag in tags:
                    if data.get(tag) is None:
                        data[tag] = Series()
                    data[tag].add(step=e.step, val=v.simple_value)

    for tag in data:
        data[tag].verify()
        steps = data[tag].steps

    for tag in tags:
        if tag not in data:
            data[tag] = get_empty_series(steps)
    return data


def combine_series(series_list, use_median=False):
    """
    :param series_list: a list of `Series` assuming steps are aligned
    :return: steps, values, stds
    """
    step_sizes = [len(series.steps) for series in series_list]
    min_idx = np.argmin(step_sizes)
    steps = series_list[min_idx].steps

    # [nb_run, nb_steps]
    all_values = [series.values[:len(steps)] for series in series_list]
    if not use_median:
        values = np.mean(all_values, axis=0)
    else:
        values = np.median(all_values, axis=0)
    stds = np.std(all_values, axis=0)
    return steps, values, stds


def plot_each_tag(all_data, ax, exp_names, max_steps, tag, font_size, use_median, tags, names, max_plot_steps=None):
    data = {}
    min_steps = math.inf
    for exp_name in exp_names:
        steps, means, stds = combine_series([run_data[tag] for run_data in all_data[exp_name]],
                                            use_median=use_median)
        data[exp_name] = (steps, means, stds)
        if steps[-1] < min_steps:
            min_steps = steps[-1]
    # Use Plot until max_steps + 10k
    for exp_name in exp_names:
        steps, means, stds = data[exp_name]
        plot_steps = min(min_steps, max_plot_steps) if max_plot_steps is not None else min_steps
        new_steps, new_means, new_stds = [], [], []
        for step, mean, std in zip(steps, means, stds):
            new_steps.append(step)
            new_means.append(mean)
            new_stds.append(std)
            if step <= plot_steps:
               new_steps.append(step)
               new_means.append(mean)
               new_stds.append(std)
            else:
               break
        new_means = np.array(new_means)
        new_stds = np.array(new_stds)
        line, = ax.plot(new_steps, new_means)
        line.set_label(exp_name)
        ax.fill_between(new_steps, new_means - new_stds, new_means + new_stds,
                        alpha=0.2)
    ax.set_xlabel('frames', fontsize=font_size)
    ax.set_title(names[tags.index(tag)])
    ax.legend(fontsize=20, loc='lower left')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir')
    parser.add_argument('output_dir')
    parser.add_argument('-same_canvas', action='store_true')
    parser.add_argument('-use_median', action='store_true')
    parser.add_argument('-tags', nargs='*')
    parser.add_argument('-names', nargs='*')
    parser.add_argument('-max_steps', default=None, type=int)
    parser.add_argument('-use_min', action='store_true')
    args = parser.parse_args()
    tags = args.tags
    names = args.names
    assert len(tags) == len(names)
    for name, tag in zip(names, tags):
        print('Plot {} ({})'.format(name, tag))
    exp_names = [exp_name for exp_name in sorted(os.listdir(args.exp_dir)) if
                 os.path.isdir(os.path.join(args.exp_dir, exp_name))]
    exp_names = sorted(exp_names)
    print('We have {} experiments'.format(len(exp_names)))

    all_data = {}
    all_tags = []
    for exp_name in exp_names:
        all_data[exp_name] = []
        runs = os.listdir(os.path.join(args.exp_dir, exp_name))
        print('We have {} runs for {}'.format(len(runs), exp_name))
        for run in runs:
            run_data = parse_tb_event_files(os.path.join(args.exp_dir, exp_name, run), tags)
            if len(run_data) == 0:
                continue
            all_tags = list(run_data.keys())
            all_data[exp_name].append(run_data)

    # Get table
    from pandas import DataFrame
    df = DataFrame(columns=all_tags)
    max_steps = {}
    for exp_name in exp_names:
        steps, mean_vals, _ = combine_series([run_data[tags[0]] for run_data in all_data[exp_name]])
        if args.use_min:
            max_id = np.argmin(mean_vals)
        else:
            max_id = np.argmax(mean_vals)
        max_step = steps[max_id]
        max_steps[exp_name] = max_step
        print('{} max step: {}'.format(exp_name, max_step))
        for tag in all_tags:
            _, means, stds = combine_series([run_data[tag] for run_data in all_data[exp_name]])
            max_mean = means[max_id]
            max_std = stds[max_id]
            df.loc[exp_name, tag] = "{:.3f}({:.3f})".format(max_mean, max_std)
    print(df)

    # Start plotting
    if args.same_canvas:
        nb_col = 2
        nb_row = int((len(tags) + 1) / nb_col)
        matplotlib.rc('font', size=20)
        fig, axs = plt.subplots(nb_row, nb_col, figsize=(13*nb_col, 10*nb_row))
        for tag, ax in zip(all_tags, axs.reshape([-1])):
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            plot_each_tag(all_data, ax, exp_names, max_steps, tag, font_size=20, use_median=args.use_median,
                          names=names, tags=tags)
        if args.use_median:
            fig.savefig(os.path.join(args.output_dir, 'result_median.png'))
        else:
            fig.savefig(os.path.join(args.output_dir, 'result_mean.png'))
    else:
        matplotlib.rc('font', size=20)
        for tag in all_tags:
            fig, ax = plt.subplots(figsize=(8, 7))
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            plot_each_tag(all_data, ax, exp_names, max_steps, tag, font_size=20, use_median=args.use_median,
                          names=names, tags=tags, max_plot_steps=args.max_steps)
            fig.savefig(os.path.join(args.output_dir, '{}.png'.format(names[tags.index(tag)])),
                        bbox_inches='tight')


if __name__ == '__main__':
    main()
