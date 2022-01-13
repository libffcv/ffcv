from gridtools import *

@param('grid.log_dir')
@param('grid.out_file')
def main(log_dir, out_file):
    out_dir = Path(log_dir) / str(uuid4())
    out_dir.mkdir(exist_ok=True, parents=True)

    # wds = [Parameters(wd=wd) for wd in [5e-4, 1e-4, 5e-5, 1e-5]]
    # lrs = [Parameters(lr=float(lr)) for lr in np.linspace(.1, 2., 9)]

    starts = []
    res = [Parameters(min_res=224, max_res=224, val_res=312)]
    for num_epochs in [30]:
        lengths, ends = [4, 8], [num_epochs, num_epochs - 4]
        res += [Parameters(min_res=160, max_res=224, val_res=312, start_ramp=e - l,
                           end_ramp=e) for l, e in itertools.product(lengths, ends)]
    import random
    random.shuffle(res)

    base_dir = '/ssd3/' if os.path.exists('/ssd3/') else '/mnt/cfs/home/engstrom/store/ffcv/'
    archs = [
        Parameters(train_dataset=base_dir + 'train_400_0.10_90.ffcv',
                   val_dataset=base_dir + 'val_400_0.10_90.ffcv',
                   batch_size=1024,
                   arch='resnet18',
                   distributed=0,
                   logs=log_dir,
                   world_size=1),
    ]

    axes = [archs, res]

    rn18_base = 'imagenet_configs/resnet18_base.yaml'
    design_command(axes, out_dir, out_file, rn18_base)

if __name__ == '__main__':
    Section('grid', 'data related stuff').params(
        log_dir=Param(str, 'out directory', default='/mnt/cfs/home/engstrom/store/ffcv_rn18_1gpu/'),
        out_file=Param(str, 'out file', default='/mnt/cfs/home/engstrom/store/ffcv_rn18_1gpu/jobs_18.txt')
    )

    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    main()

