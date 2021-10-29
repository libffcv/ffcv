import subprocess
import tqdm
from pathlib import Path
import numpy as np
from argparse import ArgumentParser
from fastargs import Param, Section
from fastargs.decorators import param
from fastargs.validation import And, OneOf
from fastargs import get_current_config
from itertools import product

Section('grid', 'data related stuff').params(
#     wds=Param(str, 'csv wds', default=),
#     lrs=Param(str, 'csv lrs', required=True),
#     epochs=Param(str, 'csv epochs', required=True),
    out_dir=Param(str, 'out directory', required=True)
)

# @param('grid.wds')
# @param('grid.lrs')
# @param('grid.epochs')
@param('grid.out_dir')
def main(wds, lrs, epochs, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    for wd, lr, num_epoch in tqdm.tqdm(list(product(wds, lrs, epochs))):
        these_args = ['./train_cifar.sh', lr, wd, num_epoch, out_dir]
        cmds = ' '.join(map(str, these_args))
        print(f'Running {cmds}')
        subprocess.run(cmds, shell=True)

if __name__ == '__main__':
    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    wds = [5e-5, 1e-5, 1e-6, 1e-4]
    lrs = [1.5, 1.25, 1.1] + list(np.linspace(0.1, 1, 10)) + list(np.linspace(0.01, 0.09, 9))
    epochs = [14, 24, 48, 96]
    print(len(wds) * len(lrs) * sum(epochs) * (40/24) / 60 / 60, 'hours!')
    main(wds, lrs, epochs)