import torch.nn as nn
import torch


class MagnetoOpticalCNN(nn.Module):
    def __init__(self, width_image, height_image, channels):
        super().__init__()
        self.network_before = nn.Sequential(
            nn.Conv2d(channels, 8, 3, 1, 1),
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.Conv2d(16, 32, 3, 1, 1),
        )
        self.network_after = nn.Sequential(
            nn.Conv2d(channels, 8, 3, 1, 1),
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.Conv2d(16, 32, 3, 1, 1),
        )
        trial_size = torch.randn((1, channels, width_image, height_image))
        trial_size = self.network_before(trial_size)
        trial_size = nn.Flatten(trial_size)
        self.shape = trial_size.shape[-1]
        self.linear_start = nn.Linear(2 * self.shape + 3, 512)
        self.linear = nn.Sequential(
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 4)  # (x,y) coordinates of refraction point, angle of incidence, angle of refraction
        )

    def forward(self, img_1, img_2, conditions):
        conditions = torch.nn.functional.one_hot(conditions, num_classes=3)
        img_1 = self.network_before(img_1)
        img_2 = self.network_after(img_2)
        in_layer = torch.cat((conditions, img_1, img_2), dim=-1)
        return self.linear(self.linear_start(in_layer))
