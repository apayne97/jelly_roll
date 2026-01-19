import torch
import torch.nn as nn
import torchvision
import numpy as np

import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid
from pydantic import BaseModel
from pathlib import Path
import json


class BaseModelConfig(BaseModel):
    model_name: str = "BaseModelConfig"

    @classmethod
    def from_json_file(cls, path: str):
        import json

        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_json_file(self, path: str):
        import json

        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=4)


# set up encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.linear1 = nn.Linear(in_features=input_dim, out_features=2 * hidden_dim)
        self.linear2 = nn.Linear(in_features=2 * hidden_dim, out_features=hidden_dim)
        self.linear3 = nn.Linear(in_features=hidden_dim, out_features=latent_dim)

    def forward(self, x):
        x = x.view(-1, self.input_dim)

        x = F.relu(self.linear1(x))

        x = F.relu(self.linear2(x))

        return self.linear3(x)


# set up decoder
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(in_features=latent_dim, out_features=hidden_dim)
        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=2 * hidden_dim)
        self.linear3 = nn.Linear(in_features=2 * hidden_dim, out_features=input_dim)

    def forward(self, z):
        z = F.relu(self.linear1(z))

        z = F.relu(self.linear2(z))

        z = self.linear3(z)  # No sigmoid: allow unbounded output

        return z.reshape(-1, 2)


class LinearAutoencoderConfig(BaseModelConfig):
    model_name: str = "LinearAutoencoder"
    input_dim: int
    hidden_dim: int
    latent_dim: int


class LinearAutoencoder(nn.Module):
    def __init__(self, config: LinearAutoencoderConfig):
        super(LinearAutoencoder, self).__init__()

        self.config = config

        self.encoder = Encoder(config.input_dim, config.hidden_dim, config.latent_dim)

        self.decoder = Decoder(config.input_dim, config.hidden_dim, config.latent_dim)

    def forward(self, x):
        z = self.encoder(x)

        return self.decoder(z)


class ModelConfigConstructor(BaseModel):
    model_config_json: str | Path

    def load_config(self) -> BaseModelConfig:
        with open(self.model_config_json, "r") as f:
            ml_model_config = json.load(f)
        match ml_model_config["model_name"]:
            case "LinearAutoencoder":
                return LinearAutoencoderConfig(**ml_model_config)
            case _:
                raise ValueError(f"Unknown model type: {ml_model_config.model_name}")


class ModelConstructor(BaseModel):
    weights_path: str | Path
    model_config_json: str | Path

    def build_model(self) -> nn.Module:
        ml_model_config = ModelConfigConstructor(
            model_config_json=self.model_config_json
        ).load_config()

        match ml_model_config.model_name:
            case "LinearAutoencoder":
                model = LinearAutoencoder(config=ml_model_config)
            case _:
                raise ValueError(f"Unknown model type: {ml_model_config.model_name}")

        if self.weights_path is not None:
            model.load_state_dict(torch.load(self.weights_path))

        return model
