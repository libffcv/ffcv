import subprocess
import itertools
from uuid import uuid4
import tqdm
import copy
from pathlib import Path
import numpy as np
from argparse import ArgumentParser
from fastargs import Param, Section
from fastargs.decorators import param
from fastargs.validation import And, OneOf
from fastargs import get_current_config
from itertools import product
import yaml

MAPPING = {
    'wd': ['training', 'weight_decay'],
    'lr': ['training', 'lr'],
    'momentum': ['training', 'momentum'],
    'label_smoothing': ['training', 'label_smoothing'],
    'blurpool': ['model', 'antialias'],
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
    'peak':['training', 'lr_peak_epoch']
}

STANDARD_CONFIG = yaml.safe_load(open('imagenet_configs/resnet18_90.yaml', 'r'))

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

# params = Parameters(lr=88, wd=99, arch='resnet50')
# print(d)
# params.override(d)
# print(d)

@param('grid.log_dir')
@param('grid.out_file')
def main(log_dir, out_file):
    out_dir = Path(log_dir) / str(uuid4())
    out_dir.mkdir(exist_ok=True, parents=True)

    wds = [Parameters(wd=wd) for wd in [1e-4]]
    lrs = [Parameters(lr=lr) for lr in [0.55, 0.65]]
    res = [Parameters(min_res=k, max_res=k, val_res=kv) for k, kv in [
        (224, 312), (160, 224), (192, 256)
    ]]
    epochs = [Parameters(epochs=90)]

    datasets = [
        Parameters(train_dataset='/mnt/cfs/home/engstrom/store/ffcv/train_350_0_100.ffcv',
                   val_dataset='/mnt/cfs/home/engstrom/store/ffcv/val_350_0_100.ffcv')
    ]

    # next:
    axes = [wds, lrs, [Parameters(logs=log_dir)], datasets, epochs, res]
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

if __name__ == '__main__':
    Section('grid', 'data related stuff').params(
        log_dir=Param(str, 'out directory', required=True),
        out_file=Param(str, 'out file', required=True)
    )

    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    main()
