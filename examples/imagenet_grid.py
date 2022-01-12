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
    'peak':['training', 'lr_peak_epoch'],
    'bn_wd':['training', 'bn_wd'],
    'mixup':['training', 'mixup_alpha'],
    'same_lambda':['training', 'mixup_same_lambda'],
    'schedule_type':['training', 'lr_schedule_type'],
    'distributed':['training', 'distributed'],
    'world_size':['dist', 'world_size'],
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


@param('grid.log_dir')
@param('grid.out_file')
def main(log_dir, out_file):
    out_dir = Path(log_dir) / str(uuid4())
    out_dir.mkdir(exist_ok=True, parents=True)

    wds = [Parameters(wd=wd) for wd in  [5e-3, 1e-3, 5e-4, 1e-4]]
    lrs = [Parameters(lr=float(lr)) for lr in np.linspace(.1, 1., 4)]
    res = [Parameters(min_res=k, max_res=k, val_res=kv) for k, kv in [
        (160, 224)
    ]]

    base_dir = '/ssd3/' if os.path.exists('/ssd3/') else '/mnt/cfs/home/engstrom/store/ffcv/'
    archs = [
        Parameters(train_dataset=base_dir + 'train_350_0_100.ffcv',
                   val_dataset=base_dir + 'val_350_0_100.ffcv',
                   batch_size=512,
                   arch='resnet50',
                   distributed=1,
                   world_size=8),
        # Parameters(train_dataset='/mnt/cfs/home/engstrom/store/ffcv/train_500_0.5_90.ffcv',
        #            val_dataset='/mnt/cfs/home/engstrom/store/ffcv/val_500_0.5_90.ffcv',
        #            batch_size=512,
        #            arch='resnet50')
    ]

    peaks = [Parameters(peak=k, schedule_type='linear') for k in [0]]
    epochs = [Parameters(epochs=k) for k in [30]]

    should_mixup = [
        Parameters(mixup=0., same_lambda=1)
    ]

    should_bn_wd = [Parameters(bn_wd=False)]

    # next:
    axes = [wds, lrs, [Parameters(logs=log_dir)], archs, epochs, res,
            should_mixup, should_bn_wd, peaks, archs]
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
    cmd = "parallel --jobs 1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
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
