Section('training.resolution_schedule',
        'How the resolution increases during training').params(
            min_resolution=Param(int, 'resolution at the first epoch', default=160),
            end_ramp=Param(int, 'At which epoch should we end increasing the resolution',
                           default=20),
            max_resolution=Param(int, 'Resolution we reach at end', default=160),
        )

Section('optimizations').params(
    label_smoothing=Param(float, 'alpha for label smoothing'),
    blurpool=Param(int, 'Whether to use blurpool layers', default=1),
    tta=Param(int, 'Whether to use test-time augmentation', default=1)
)

class TTAModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        if self.training: return self.model(x)
        return self.model(x) + self.model(ch.flip(x, dims=[3]))

@section('training.resolution_schedule')
@param('min_resolution')
@param('max_resolution')
@param('end_ramp')
def get_resolution_schedule(min_resolution, max_resolution, end_ramp):
    def schedule(epoch):
        diff = max_resolution - min_resolution
        result =  min_resolution
        result +=  min(1, epoch / end_ramp) * diff
        result = int(np.round(result / 32) * 32)
        return result
    return schedule