"""Pytorch Model."""

from torch import nn


class Model(nn.Module):

    def __init__(self, n_classes: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        layers = []
        channels = [3, 32, 64, 128, 128]
        for i in range(1, len(channels)):
            layers += [
                nn.Conv2d(channels[i - 1],
                          channels[i],
                          kernel_size=3,
                          padding='same'),
                nn.BatchNorm2d(channels[i]),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
            ]

        self.backbone = nn.Sequential(*layers)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(channels[-1], 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
