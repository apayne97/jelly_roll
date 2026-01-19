import pytest
from jelly_roll.data import JellyRoll


def test_spiral_generation():
    jelly_roll = JellyRoll(theta_max=4 * 3.14159, n_points=500, a=0.2, b=0.3)
    x, y = jelly_roll.generate_xy()
    assert len(x) == 500
    assert len(y) == 500
    assert x[0] != x[-1]  # Ensure the spiral is not a single point


def test_data_loader():
    jelly_roll = JellyRoll()
    train, val, test = jelly_roll.get_data_loaders()
    print(
        f"Train: {len(train.dataset)}, Valid: {len(val.dataset)}, Test: {len(test.dataset)}"
    )
    assert len(train.dataset) == 600
    assert len(val.dataset) == 200
    assert len(test.dataset) == 200
