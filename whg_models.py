from __future__ import annotations

import torch as th
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class WHGCNN(BaseFeaturesExtractor):
    """Configurable NatureCNN-style feature extractor for WHG.

    Keep this in a separate module and import it only inside train.main().
    That prevents Windows spawn environment workers from importing PyTorch/CUDA.
    """

    def __init__(
        self,
        observation_space,
        features_dim: int = 512,
        channels: tuple[int, int, int] = (32, 64, 64),
    ):
        super().__init__(observation_space, features_dim)

        c1, c2, c3 = map(int, channels)
        n_input_channels = int(observation_space.shape[0])

        self.cnn = nn.Sequential(
            nn.Conv2d(
                n_input_channels,
                c1,
                kernel_size=8,
                stride=4,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                c1,
                c2,
                kernel_size=4,
                stride=2,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                c2,
                c3,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Determine flatten size without hard-coding the input resolution.
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = int(self.cnn(sample).shape[1])

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, int(features_dim)),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
