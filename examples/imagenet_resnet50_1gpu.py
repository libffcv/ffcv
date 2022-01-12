from gridtools import *

@param('grid.log_dir')
@param('grid.out_file')
def main(log_dir, out_file):
    out_dir = Path(log_dir) / str(uuid4())
    out_dir.mkdir(exist_ok=True, parents=True)

    wds = [Parameters(wd=wd) for wd in [1e-4, 5e-5]]
    lrs = [Parameters(lr=float(lr)) for lr in np.linspace(.1, 2., 9)]
    res = [Parameters(min_res=k, max_res=k, val_res=kv) for k, kv in [
        (160, 224) #, (192, 256)
    ]]

    base_dir = '/ssd3/' if os.path.exists('/ssd3/') else '/mnt/cfs/home/engstrom/store/ffcv/'
    archs = [
        Parameters(train_dataset=base_dir + 'train_500_0.5_90.ffcv',
                   val_dataset=base_dir + 'val_500_0.5_90.ffcv',
                   batch_size=512,
                   arch='resnet50',
                   distributed=0,
                   world_size=1),
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
    design_command(axes, out_dir, out_file)

if __name__ == '__main__':
    Section('grid', 'data related stuff').params(
        log_dir=Param(str, 'out directory', default='/mnt/cfs/home/engstrom/store/ffcv_rn50_1gpu/'),
        out_file=Param(str, 'out file', default='/tmp/jobs_50.txt')
    )

    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    main()
