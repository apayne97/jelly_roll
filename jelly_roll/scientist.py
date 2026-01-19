from pathlib import Path

import click
import numpy as np

from jelly_roll.data import JellyRoll
from jelly_roll.trainer import train_model
from jelly_roll.plotting import plot_jelly_roll

from pydantic import BaseModel
from typing import Union
from itertools import product
import torch


def permute_dict(d: dict) -> list[dict]:
    """Generate all combinations of parameter values from a dictionary.

    Args:
        d (dict): Dictionary with parameter names as keys and lists of possible values.

    Returns:
        list[dict]: List of dictionaries with all combinations of parameter values.
    """
    keys = d.keys()
    values = d.values()
    combinations = product(*values)
    return [dict(zip(keys, combo)) for combo in combinations]


class Experiment(BaseModel):
    """Experiment configuration model.

    Attributes:
        description (str): Description of the experiment.
    """

    unique_id: str
    description: str
    model_type: str
    experiment_config: dict
    epochs: int = 40
    learning_rate: float = 1e-3

    def build_models(self):
        """Build the model based on the configuration.

        Returns:
            model: Instantiated model.
        """
        match self.model_type:
            case "LinearAutoencoder":
                from jelly_roll.models import LinearAutoencoder, LinearAutoencoderConfig

                configs = []
                for config in permute_dict(self.experiment_config):
                    configs.append(LinearAutoencoderConfig(**config))
                return [LinearAutoencoder(config) for config in configs]

    @classmethod
    def from_yaml(cls, path: str | Path):
        """Load experiment configuration from a YAML file.

        Args:
            path (str): Path to the YAML file.
        Returns:
            Experiment: Loaded experiment configuration.
        """
        import yaml

        with open(path, "r") as file:
            data = yaml.safe_load(file)
        return cls(**data)

    def to_yaml(self, path: str | Path):
        """Save experiment configuration to a YAML file.

        Args:
            path (str): Path to save the YAML file.
        """
        import yaml

        with open(path, "w") as file:
            yaml.safe_dump(self.model_dump(), file)


@click.group("scientist")
def scientist():
    """Scientist CLI for running experiments."""
    pass


@scientist.command("run-experiment")
@click.argument("input-yaml", type=click.Path(exists=True))
def run_experiment(input_yaml):
    """Run a full experiment: train model and plot results."""
    click.echo("Starting experiment...")
    experiment = Experiment.from_yaml(input_yaml)
    output_dir = Path("experiments") / experiment.unique_id
    output_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"Experiment ID: {experiment.unique_id}")
    click.echo(f"Description: {experiment.description}")
    jr = JellyRoll()
    plot_jelly_roll(jr)
    train, val, test = jr.get_data_loaders()
    for model in experiment.build_models()[:1]:
        train_loss_list, val_loss_list, model = train_model(
            train,
            val,
            model,
            epochs=experiment.epochs,
            learning_rate=experiment.learning_rate,
        )
        click.echo(f"Trained model: {model}")
        torch.save(model.state_dict(), output_dir / f"{model.__class__.__name__}.pth")
        np.save(output_dir / f"train_loss.npy", np.array(train_loss_list))
        np.save(output_dir / f"val_loss.npy", np.array(val_loss_list))
        model.config.to_json_file(
            output_dir / f"{model.__class__.__name__}_config.json"
        )

    click.echo("Experiment completed.")


@scientist.command("assess-experiment")
@click.argument("experiment-dir", type=click.Path(exists=True))
def assess_experiment(experiment_dir):
    """Analyze the results of an experiment"""
    from jelly_roll.plotting import plot_loss

    experiment_dir = Path(experiment_dir)

    train_loss = np.load(experiment_dir / "train_loss.npy")
    val_loss = np.load(experiment_dir / "val_loss.npy")
    plot_loss(train_loss, val_loss)

    from jelly_roll.models import ModelConstructor

    model_constructor = ModelConstructor(
        weights_path=experiment_dir / "LinearAutoencoder.pth",
        model_config_json=experiment_dir / "LinearAutoencoder_config.json",
    )
    model = model_constructor.build_model()

    jr = JellyRoll()
    data = jr.generate()
    train, val, test = jr.get_data_loaders()

    # Get a batch from the validation set
    X_val = next(iter(val))

    # Get reconstructions and denormalize back to original scale
    with torch.no_grad():
        model.eval()
        # Reconstruct on validation set
        reconstruction = model(X_val)
        reconstruction_np = reconstruction.detach().cpu().numpy()
        X_val_np = X_val.detach().cpu().numpy()

    # denormalize
    reconstruction_denorm = jr.denormalize(reconstruction_np)
    X_val_denorm = jr.denormalize(X_val_np)

    from jelly_roll.plotting import plot_reconstruction

    plot_reconstruction(jr, X_val_denorm, reconstruction_denorm)
