import matplotlib.pyplot as plt
import seaborn as sns
import jelly_roll.data as jr
from pathlib import Path
import torch


def plot_jelly_roll(jelly_roll: jr.JellyRoll):
    """Plot the jelly roll shape.

    Args:
        jelly_roll (jr.JellyRoll): JellyRoll instance to plot.
    """
    x, y = jelly_roll.generate_xy()
    plt.figure(figsize=(8, 8))
    sns.set_style("whitegrid")
    plt.plot(x, y, color="blue", linewidth=2)
    plt.title("Jelly Roll Shape", fontsize=16)
    plt.xlabel("X-axis", fontsize=14)
    plt.ylabel("Y-axis", fontsize=14)
    plt.axis("equal")
    plt.savefig("jelly_roll.png")
    plt.close()


def plot_loss(train, val, test=None, output_dir="."):
    """Plot training, validation, and test loss curves.

    Args:
        train (list): Training loss values.
        val (list): Validation loss values.
        test (list): Test loss values.
    """
    output_dir = Path(output_dir)
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    plt.plot(train, label="Training Loss", color="blue")
    plt.plot(val, label="Validation Loss", color="orange")
    if test is not None:
        plt.plot(test, label="Test Loss", color="green")
    plt.title("Loss Curves", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend()
    plt.savefig("loss_curves.png")
    plt.close()


def plot_reconstruction(jr: jr.JellyRoll, X_val, reconstruction):
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.set_style("whitegrid")
    spiral_x, spiral_y = jr.generate_xy()
    ax.plot(
        spiral_x,
        spiral_y,
        color="green",
        linewidth=2,
        label="True spiral",
        alpha=0.7,
    )
    ax.scatter(
        reconstruction[:, 0],
        reconstruction[:, 1],
        color="red",
        s=20,
        alpha=0.6,
        label="Reconstructions",
    )
    ax.scatter(
        X_val[:, 0],
        X_val[:, 1],
        color="blue",
        s=20,
        alpha=0.6,
        label="Input validation data",
    )
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("Autoencoder Reconstruction vs Input Data")
    plt.tight_layout()
    plt.savefig("reconstruction.png")
    plt.close()
