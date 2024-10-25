
class Sampler:
    def __init__(self, **kwargs):
        pass

    def __call__(self, logits, random=False):
        # converts logits into a sampleable goal from the lower policy goal space
        # if random is true, uses logits for batch size and returns a random sample
        raise NotImplementedError
