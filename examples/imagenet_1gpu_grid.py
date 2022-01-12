from gridtools import *

STANDARD_CONFIG = yaml.safe_load(open('imagenet_configs/resnet50_1gpu.yaml', 'r'))

@param('grid.log_dir')
@param('grid.out_file')
def main(log_dir, out_file):
    out_dir = Path(log_dir) / str(uuid4())
    out_dir.mkdir(exist_ok=True, parents=True)

    wds = [Parameters(wd=wd) for wd in [5e-4, 1e-4]]
    lrs = [Parameters(lr=float(lr)) for lr in np.linspace(.1, 2., 8)]
    res = [Parameters(min_res=k, max_res=k, val_res=kv) for k, kv in [
        (160, 224) #, (192, 256)
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
    epochs = [Parameters(epochs=k) for k in [35]]

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
