import subprocess
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
    fig, ax = plt.subplots()

    for name, group in logs.groupby('resolution.max_res'):
        group.plot.line(x='relative_time', y='top_5', label=name, ax=ax)
    plt.show()
    print(logs)
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
