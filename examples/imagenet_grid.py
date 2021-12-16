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
    'end_ramp': ['resolution', 'end_ramp'],
    'val_res': ['validation', 'resolution'],
    'logs': ['logging', 'folder'],
    'batch_size':['training', 'batch_size'],
    'peak':['training', 'lr_peak_epoch']
}

STANDARD_CONFIG = yaml.safe_load(open('imagenet_configs/juiced.yaml', 'r'))

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

    constant_params = Parameters(wd=4e-5, lr=0.2, label_smoothing=.1,
                                     blurpool=0, arch='resnet18')
    # wd, lr, min_res, max_res, end_ramp, momentum, label_smoothing, blurpool
    # wds = [4e-5]
    # lrs = [0.2]
    max_ress = [Parameters(max_res=k, logs=str(log_dir),
                           val_res=(k + 64)) for k in [160]]
    
    min_ress = [Parameters(min_res=k) for k in [96]]
    end_ramps = [Parameters(end_ramp=k, epochs=k+delta) for k, delta in itertools.product([15, 20, 30], [5, 10, 15, 20])]
    peaks = [Parameters(lr_peak_epoch=k) for k in [3, 5, 7, 9]]
    blurpools = [Parameters(blurpool=0)]
    wds = [Parameters(wd=1e-4)]

    # next:
    bslr = [Parameters(batch_size=256 * k, lr=0.1 * k) for k in [4, 8]]
    axes = [min_ress, end_ramps, max_ress, blurpools, bslr, wds, peaks]

    out_write = []
    configs = list(itertools.product(*axes))
    # configs = [[Parameters(lr=0.2, blurpool=0, wd=4e-5, end_ramp=0,
    #            batch_size=512, min_res=160, max_res=160, val_res=160+64,
    #            logs=str(log_dir), epochs=100)]] + configs

    for these_settings in configs:
        d = copy.deepcopy(STANDARD_CONFIG)
        constant_params.override(d)
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

    # for wd, lr, num_epoch in tqdm.tqdm(list(product(wds, lrs, epochs))):
    #     these_args = ['./train_cifar.sh', lr, wd, num_epoch, out_dir]
    #     cmds = ' '.join(map(str, these_args))
    #     print(f'Running {cmds}')
    #     subprocess.run(cmds, shell=True)

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
#     wds = [1e-3, 1e-5, 5e-5, 1e-4, 5e-4]
#     lrs = [3, 2, 1.5, 1.25, 1.1] + list(np.linspace(0.1, 1, 10)) + list(np.linspace(0.01, 0.09, 9))
#     epochs = [6, 12, 36, 72]
#     print(len(wds) * len(lrs) * sum(epochs) * (40/24) / 60 / 60, 'hours!')
    main()
