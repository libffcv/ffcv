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