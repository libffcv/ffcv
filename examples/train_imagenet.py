class TTAModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        if self.training: return self.model(x)
        return self.model(x) + self.model(ch.flip(x, dims=[3]))