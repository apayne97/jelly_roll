import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import click
from tqdm import tqdm
import numpy as np
from pydantic import BaseModel
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"


@click.group()
def trainer():
    pass

class TrainingResults(BaseModel):

    class Config:
        arbitrary_types_allowed = True

    train_loss: np.ndarray
    val_loss: np.ndarray
    model: nn.Module
    val_X: np.ndarray = None
    val_X_reconstructed: np.ndarray = None

    def write_results(self, output_dir: Path = Path("./results")):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "train_loss.npy", self.train_loss)
        np.save(output_dir / "val_loss.npy", self.val_loss)

        torch.save(self.model.state_dict(), output_dir / f"model_weights.pth")
        self.model.config.to_json_file(output_dir / "model_config.json")
        if self.val_X is not None:
            np.save(output_dir / "val_X.npy", self.val_X)
        if self.val_X_reconstructed is not None:
            np.save(output_dir / "val_X_reconstructed.npy", self.val_X_reconstructed)

    @classmethod
    def from_results_dir(cls, results_dir: Path):
        train_loss = np.load(results_dir / "train_loss.npy")
        val_loss = np.load(results_dir / "val_loss.npy")

        val_X_path = results_dir / "val_X.npy"
        val_X_reconstructed_path = results_dir / "val_X_reconstructed.npy"
        val_X = np.load(val_X_path) if val_X_path.exists() else None
        val_X_reconstructed = np.load(val_X_reconstructed_path) if val_X_reconstructed_path.exists() else None

        from jelly_roll.models import ModelConstructor
        model_constructor = ModelConstructor(weights_path=results_dir / "model_weights.pth",
                                                model_config_json=results_dir / "model_config.json")
        model = model_constructor.build_model()
        return cls(train_loss=train_loss,
                   val_loss=val_loss,
                   model=model,
                   val_X=val_X,
                   val_X_reconstructed=val_X_reconstructed)


def train_model(
    train_dl, valid_dl, model, epochs: int = 40, learning_rate: float = 1e-3
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_list = []
    val_loss_list = []
    val_X_list = []
    val_X_reconstructed = []

    for epoch in tqdm(range(epochs + 1)):

        if epoch > 0:
            model.train()
            train_running_loss = 0.0

            for X_train in train_dl:
                X_train = X_train.to(device)

                optimizer.zero_grad()

                reconstruction = model(X_train)

                loss = criterion(reconstruction, X_train)
                loss.backward()

                train_running_loss += loss.item()
                optimizer.step()

            train_loss = train_running_loss / len(train_dl)
            train_loss_list.append(train_loss)

        with torch.no_grad():
            model.eval()
            valid_running_loss = 0.0

            epoch_val_x_list = []
            epoch_val_x_reconstructed = []
            for X_val in valid_dl:
                X_val = X_val.to(device)

                reconstruction = model(X_val)

                loss = criterion(reconstruction, X_val)

                valid_running_loss += loss.item()
                X_val_np = X_val.detach().cpu().numpy()
                epoch_val_x_list.extend(X_val_np)
                epoch_val_x_reconstructed.extend(reconstruction.detach().cpu().numpy())

            val_X_list.append(epoch_val_x_list)
            val_X_reconstructed.append(epoch_val_x_reconstructed)
            val_loss = valid_running_loss / len(valid_dl)
            val_loss_list.append(val_loss)

    return TrainingResults(train_loss=np.array(train_loss_list), val_loss=np.array(val_loss_list), model=model, val_X=np.array(val_X_list), val_X_reconstructed=np.array(val_X_reconstructed))
