import torch
import torch.nn as nn

from .discriminator import Discriminator

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()

        self.discriminators = nn.ModuleList(
            [Discriminator() for _ in range(3)]
        )
        
        self.downsample = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        scores = list()

        for layer in self.discriminators:
            score = layer(x)
            scores.append(score)

            x = self.downsample(x)

        return scores

if __name__ == '__main__':
    model = MultiScaleDiscriminator()

    x = torch.randn(3, 1, 16000)

    scores = model(x)
    for score in scores:
        print(score.shape)
