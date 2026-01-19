import numpy as np
from pydantic import BaseModel
import torch
from torch.utils.data import DataLoader, random_split


def spiral(theta_max=6 * np.pi, n_points=1000, a=0.1, b=0.2) -> np.ndarray:
    """Return x, y coordinates for an Archimedean spiral r = a + b*theta.

    Args:
        theta_max (float): maximum angle in radians
        n_points (int): number of points along the curve
        a (float): initial radius offset
        b (float): radial growth per radian

    Returns:
        np.ndarray: array of shape (n_points, 2) with x and y coordinates
    """
    theta = np.linspace(0, theta_max, int(n_points))
    r = a + b * theta
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.vstack((x, y)).T


class SplitConfig(BaseModel):
    """Configuration for dataset splitting.

    Attributes:
        train_fraction (float): Fraction of data to use for training.
        val_fraction (float): Fraction of data to use for validation.
    """

    train_fraction: float = 0.6
    val_fraction: float = 0.2
    test_fraction: float = 0.2


class JellyRoll(BaseModel):
    """Jelly Roll shape generator.

    Attributes:
        theta_max (float): maximum angle in radians
        n_points (int): number of points along the curve
        a (float): initial radius offset
        b (float): radial growth per radian
    """

    theta_max: float = 6 * np.pi
    n_points: int = 1000
    a: float = 0.1
    b: float = 0.2

    def generate(self):
        """Generate the jelly roll shape coordinates.

        Returns:
            tuple: (x, y) arrays
        """
        return spiral(self.theta_max, self.n_points, self.a, self.b)

    def generate_xy(self):
        """Generate the jelly roll shape coordinates.

        Returns:
            tuple: (x, y) arrays
        """
        data = self.generate()
        x = data[:, 0]
        y = data[:, 1]
        return x, y

    def get_data_loaders(
        self, split_config: SplitConfig = None, batch_size: int = 32
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Generate data loaders for training, validation, and testing.

        Args:
            split_config (SplitConfig): Configuration for dataset splitting.
            batch_size (int): Batch size for data loaders.
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        data = self.generate()

        if split_config is None:
            split_config = SplitConfig()

        # normalize
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        data_normalized = 2 * (data - data_min) / (data_max - data_min) - 1

        data_tensor = torch.tensor(data_normalized, dtype=torch.float32)
        train_data, valid_test_data = random_split(
            data_tensor,
            [
                int(self.n_points * split_config.train_fraction),
                int(
                    self.n_points
                    * (split_config.val_fraction + split_config.test_fraction)
                ),
            ],
        )
        valid_data, test_data = random_split(
            valid_test_data,
            [
                int(self.n_points * split_config.val_fraction),
                int(self.n_points * split_config.test_fraction),
            ],
        )
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """Denormalize data from [-1, 1] back to original scale.

        Args:
            data (np.ndarray): Normalized data of shape (n_points, 2).

        Returns:
            np.ndarray: Denormalized data of shape (n_points, 2).
        """
        original_data = self.generate()
        data_min = original_data.min(axis=0)
        data_max = original_data.max(axis=0)
        denormalized_data = 0.5 * (data + 1) * (data_max - data_min) + data_min
        return denormalized_data
