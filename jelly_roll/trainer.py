import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import click
from tqdm import tqdm
import numpy as np
from pydantic import BaseModel

device = "cuda" if torch.cuda.is_available() else "cpu"


@click.group()
def trainer():
    pass

class TrainingResults(Base)

def train_model(
    train_dl, valid_dl, model, epochs: int = 40, learning_rate: float = 1e-3
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_list = []
    val_loss_list = []

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
            print(f"Epoch {epoch} | Training Loss: {train_loss:.3f}")

        with torch.no_grad():
            model.eval()
            valid_running_loss = 0.0

            for X_val in valid_dl:
                X_val = X_val.to(device)

                reconstruction = model(X_val)

                loss = criterion(reconstruction, X_val)

                valid_running_loss += loss.item()

            val_loss = valid_running_loss / len(valid_dl)
            val_loss_list.append(val_loss)
    return np.array(train_loss_list), np.array(val_loss_list), model
