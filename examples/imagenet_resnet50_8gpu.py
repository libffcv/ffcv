from gridtools import *

@param('grid.log_dir')
@param('grid.out_file')
def main(log_dir, out_file):
    out_dir = Path(log_dir) / str(uuid4())
    out_dir.mkdir(exist_ok=True, parents=True)

    starts = []
    wds = [Parameters(wd=k) for k in [2.5e-4/2]]
    lrs = [Parameters(lr=float(k)) for k in np.linspace(0.1, 1.5, 4)]
    base_dir = '/home/ubuntu/' if os.path.exists('/home/ubuntu/') else '/mnt/cfs/home/engstrom/store/ffcv/'
    archs = [
        Parameters(train_dataset=base_dir + 'train_400_0.10_90.ffcv',
                   val_dataset=base_dir + 'val_400_0.10_90.ffcv',
                   batch_size=512,
                   arch='resnet50',
                   distributed=1,
                   logs=log_dir,
                   world_size=8),
    ]

    axes = [archs, wds, lrs]

    rn18_base = 'imagenet_configs/resnet50_base.yaml'
    design_command(axes, out_dir, out_file, rn18_base, cuda_preamble="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7", jobs=1)

if __name__ == '__main__':
    Section('grid', 'data related stuff').params(
        log_dir=Param(str, 'out directory', default=str(Path('~/store/ffcv_rn50_8gpu/').expanduser())),
        out_file=Param(str, 'out file', default=str(Path('~/store/ffcv_rn50_8gpu/jobs_18.txt').expanduser()))
    )

    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    main()

