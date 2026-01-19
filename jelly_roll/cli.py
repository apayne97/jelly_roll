import click
import jelly_roll.plotting as pl
from jelly_roll.trainer import trainer


@click.group()
def cli():
    pass


@cli.command("plot-jelly-roll")
def plot_jelly_roll():
    """Plot the jelly roll shape."""
    jelly_roll = pl.jr.JellyRoll()
    pl.plot_jelly_roll(jelly_roll)


cli.add_command(trainer)

from jelly_roll.scientist import run_experiment, assess_experiment

cli.add_command(run_experiment)
cli.add_command(assess_experiment)
