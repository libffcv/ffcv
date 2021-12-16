import subprocess
from collections import OrderedDict, defaultdict
from uuid import uuid4
import matplotlib as mpl
mpl.use('module://imgcat')
import matplotlib.pyplot as plt

import tqdm
import pandas as pd
import copy
from pathlib import Path
import numpy as np
from argparse import ArgumentParser
from fastargs import Param, Section
from fastargs.decorators import param
from fastargs.validation import And, OneOf
from fastargs import get_current_config
from itertools import product
from fastargs.dict_utils import NestedNamespace
from imagenet_grid import Parameters
import yaml
import torch as ch

Section('grid', 'data related stuff').params(
    log_dir=Param(str, 'out directory', required=True),
)

@param('grid.log_dir')
def main(log_dir):
    extractor = Parameters()
    print(log_dir)
    all_logs = []
    for p in Path(log_dir).glob('*.log'):
        base_path = str(p).split('.log')[0]
        uid = base_path.split('/')[-1]

        config_path = base_path + '-params.json'

        conf = pd.read_json(config_path, lines=True)
        logs = pd.read_json(p, lines=True)
        logs['uid'] = uid
        conf['uid'] = uid
        logs = logs.merge(conf, on='uid')
        all_logs.append(logs)

    logs = pd.concat(all_logs)
    logs['delta'] = logs['training.epochs'] - logs['resolution.end_ramp']
    def plot_all(x='epoch', by_col=None):
        fig, ax = plt.subplots()
        print(by_col)
        labeled = {}
        if by_col is not None:
            colors = ['red', 'blue', 'orange', 'purple', 'yellow', 'green', 'cyan']
            mapping = {}
            for i, uu in enumerate(logs[by_col].unique()):
                mapping[uu] = colors[i]
            labeled = defaultdict(lambda:False)

        for name, group in logs.groupby('uid'):
            kwargs = {}
            if by_col is not None:
                value = group.iloc[0][by_col]
                row_color = mapping[value]

                kwargs['color'] = row_color
                if not labeled[value]:
                    labeled[value] = True
                    kwargs['label'] = value
                else:
                    kwargs['label'] = False
                    kwargs['legend'] = False

            group.plot.line(x=x, y='top_1', ax=ax, **kwargs)

        plt.show()

    plot_all(x='epoch')
    plot_all(x='relative_time')

    for chungus in ['delta', 'training.lr', 'training.weight_decay',
                    'resolution.max_res', 'resolution.min_res',
                    'training.lr_peak_epoch']:
        plot_all(x='relative_time', by_col=chungus)

    plot_all()

    last = logs.sort_values('epoch').groupby('uid').last()
    keep = ['relative_time', 'epoch', 'top_1', 'top_5', 'training.batch_size',
    'resolution.max_res', 'resolution.min_res', 'model.antialias',
    'resolution.end_ramp', 'training.weight_decay']
    last = last[keep].sort_values('top_5')
    print(logs)
    print(last)
    import pdb; pdb.set_trace()
    print(logs)

if __name__ == '__main__':
    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    main()
