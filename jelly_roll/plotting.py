import matplotlib.pyplot as plt
import seaborn as sns
import jelly_roll.data as jr
from pathlib import Path
import numpy as np


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
    plt.savefig(output_dir / "loss_curves.png")
    plt.close()


def plot_reconstruction(jr: jr.JellyRoll, X_val, reconstruction, output_dir="."):
    output_dir = Path(output_dir)
    num_epochs = X_val.shape[0]

    for epoch in range(num_epochs):
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
            reconstruction[epoch, :, 0],
            reconstruction[epoch, :, 1],
            color="red",
            s=20,
            alpha=0.6,
            label="Reconstructions",
        )
        ax.scatter(
            X_val[epoch, :, 0],
            X_val[epoch, :, 1],
            color="blue",
            s=20,
            alpha=0.6,
            label="Input validation data",
        )
        ax.set_aspect("equal")
        ax.legend()
        ax.set_title(f"Autoencoder Reconstruction vs Input Data (Epoch {epoch})")
        plt.tight_layout()
        plt.savefig(output_dir / f"reconstruction_epoch_{epoch}.png")
        plt.close()


def plot_reconstruction_by_epoch(jr: jr.JellyRoll, X_val, reconstruction, output_dir="."):
    output_dir = Path(output_dir)
    num_epochs = X_val.shape[0]

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.set_style("whitegrid")

    # Plot true spiral
    spiral_x, spiral_y = jr.generate_xy()
    ax.plot(
        spiral_x,
        spiral_y,
        color="green",
        linewidth=2,
        label="True spiral",
        alpha=0.7,
    )

    # Create color map for epochs
    colors = plt.cm.viridis(np.linspace(0, 1, num_epochs))

    for epoch in range(num_epochs):
        ax.scatter(
            reconstruction[epoch, :, 0],
            reconstruction[epoch, :, 1],
            color=colors[epoch],
            s=20,
            alpha=0.6,
            label=f"Reconstruction (Epoch {epoch})" if epoch % 10 == 0 else "",
        )

    # Plot input validation data from final epoch for reference
    ax.scatter(
        X_val[-1, :, 0],
        X_val[-1, :, 1],
        color="blue",
        s=20,
        alpha=0.6,
        marker="x",
        label="Input validation data",
    )

    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("Autoencoder Reconstruction Across Epochs")
    plt.tight_layout()
    plt.savefig(output_dir / "reconstruction_all_epochs.png")
    plt.close()

def plot_reconstruction_with_lines(jr: jr.JellyRoll, X_val, reconstruction, output_dir="."):
    output_dir = Path(output_dir)
    num_epochs = X_val.shape[0]

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.set_style("whitegrid")

    # Plot true spiral
    spiral_x, spiral_y = jr.generate_xy()
    ax.plot(
        spiral_x,
        spiral_y,
        color="green",
        linewidth=2,
        label="True spiral",
        alpha=0.7,
    )

    # Create color map for epochs
    colors = plt.cm.viridis(np.linspace(0, 1, num_epochs))

    for epoch in range(num_epochs):
        # Plot scatter points
        ax.scatter(
            reconstruction[epoch, :, 0],
            reconstruction[epoch, :, 1],
            color=colors[epoch],
            s=20,
            alpha=0.6,
            label=f"Reconstruction (Epoch {epoch})" if epoch % 10 == 0 else "",
        )

    # Draw lines across epochs
    for point_idx in range(0, reconstruction.shape[1]):
        ax.plot(
            reconstruction[:, point_idx, 0],
            reconstruction[:, point_idx, 1],
            color="black",
            linewidth=1,
            alpha=1,
        )

    # Plot input validation data from final epoch for reference
    ax.scatter(
        X_val[-1, :, 0],
        X_val[-1, :, 1],
        color="blue",
        s=20,
        alpha=0.6,
        marker="x",
        label="Input validation data",
    )

    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("Autoencoder Reconstruction Across Epochs")
    plt.tight_layout()
    plt.savefig(output_dir / "reconstruction_all_epochs.png")
    plt.close()

