import subprocess
import os
import itertools
from uuid import uuid4
import tqdm
import copy
from pathlib import Path
import numpy as np
from argparse import ArgumentParser
from fastargs import Param, Section
from fastargs.decorators import param
from fastargs import get_current_config
import yaml

MAPPING = {
    'wd': ['training', 'weight_decay'],
    'lr': ['lr', 'lr'],
    'momentum': ['training', 'momentum'],
    'label_smoothing': ['training', 'label_smoothing'],
    'epochs': ['training', 'epochs'],
    'arch': ['model', 'arch'],
    'min_res': ['resolution', 'min_res'],
    'max_res': ['resolution', 'max_res'],
    'train_dataset': ['data', 'train_dataset'],
    'val_dataset': ['data', 'val_dataset'],
    'end_ramp': ['resolution', 'end_ramp'],
    'val_res': ['validation', 'resolution'],
    'logs': ['logging', 'folder'],
    'batch_size':['training', 'batch_size'],
    'peak':['lr', 'lr_peak_epoch'],
    'schedule_type':['lr', 'lr_schedule_type'],
    'distributed':['training', 'distributed'],
    'world_size':['dist', 'world_size'],
}

class Parameters():
    def __init__(self, mapping=MAPPING, **kwargs):
        self.kwargs = kwargs
        self.mapping = mapping
        for k in kwargs.keys():
            assert k in mapping.keys(), k

    def parent_nab(self, k, d):
        path = self.mapping[k]
        pre = path[:-1]
        final = path[-1]
        tunnel = d
        for pre_key in pre:
            tunnel = tunnel[pre_key]

        return tunnel, final

    def override(self, d):
        for k, v in self.kwargs.items():
            tunnel, final = self.parent_nab(k, d)
            tunnel[final] = v

    def read(self, d, keys):
        ret = {}
        for k in keys:
            tunnel, final = self.parent_nab(k, d)
            ret[k] = tunnel[final]

        return ret

STANDARD_CONFIG = yaml.safe_load(open('imagenet_configs/base.yaml', 'r'))


def design_command(axes, out_dir, out_file):
    out_write = []
    configs = list(itertools.product(*axes))

    for these_settings in configs:
        d = copy.deepcopy(STANDARD_CONFIG)
        for settings in these_settings:
            settings.override(d)

        uid = str(uuid4())
        d['uid'] = uid
        out_conf = out_dir / (uid + '.yaml')
        yaml.safe_dump(d, open(out_conf, 'w+'))
        out_write.append(str(out_conf))

    print(' --- out writes --- ')
    print('jobs', len(out_write))
    to_write = '\n'.join(out_write)
    open(out_file, 'w+').write(to_write)
    cmd = "parallel -j9 CUDA_VISIBLE_DEVICES='$(({%} - 1))'"
    cmd = f'cat {out_file} | ' + cmd + ' python train_imagenet.py --config-file'
    cmd = cmd + ' {}'
    print(' --- logs in --- ')
    print(out_dir)
    print(' --- run command --- ')
    print(cmd)
