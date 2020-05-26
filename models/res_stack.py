import torch.nn as nn

class ResStack(nn.Module):
    def __init__(self, channel):
        super(ResStack, self).__init__()

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(),
                nn.utils.weight_norm(nn.Conv1d(channel, channel,
                    kernel_size=3, dilation=3**i, padding=3**i)),
                nn.LeakyReLU(),
                nn.utils.weight_norm(nn.Conv1d(channel, channel,
                    kernel_size=3, dilation=1, padding=1)),
            )
            for i in range(4)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x
