from jelly_roll.scientist import Experiment, run_experiment
import pytest


@pytest.fixture
def yaml_file(tmp_path):
    # Create a temporary YAML file for testing
    yaml_content = """
    name: Test Experiment
    description: This is a test experiment.
    """
    yaml_path = tmp_path / "experiment.yaml"
    yaml_path.write_text(yaml_content)
    return str(yaml_path)


def test_experiment_from_yaml(yaml_file):
    # Load the experiment from the YAML file
    experiment = Experiment.from_yaml(str(yaml_file))

    # Assert the fields are correctly loaded
    assert experiment.name == "Test Experiment"
    assert experiment.description == "This is a test experiment."


def test_experiment_yaml_roundtrip(yaml_file, tmp_path):
    # Load the experiment from the YAML file
    experiment = Experiment.from_yaml(str(yaml_file))

    experiment.to_yaml(tmp_path / "output_experiment.yaml")

    # Load the experiment again from the new YAML file
    loaded_experiment = Experiment.from_yaml(tmp_path / "output_experiment.yaml")
    assert loaded_experiment == experiment


def test_permute_code():
    from jelly_roll.scientist import permute_dict

    config = {
        "input_dim": [2],
        "latent_dim": [1, 2],
        "hidden_dim": [[4, 4]],
    }
    permutations = list(permute_dict(config))
    assert len(permutations) == 2
    assert permutations[0] == {"input_dim": 2, "latent_dim": 1, "hidden_dim": [4, 4]}
    assert permutations[1] == {"input_dim": 2, "latent_dim": 2, "hidden_dim": [4, 4]}


def test_config_generation():
    experiment_config = {
        "input_dim": [2],
        "latent_dim": [1, 2, 3],
        "hidden_dim": [4, 8, 12],
    }
    experiment = Experiment(
        description="Test generating model from config",
        model_type="LinearAutoencoder",
        experiment_config=experiment_config,
    )
    from jelly_roll.scientist import permute_dict

    configs = permute_dict(experiment_config)

    models = experiment.build_models()
    for i, model in enumerate(models):
        config = list(configs)[i]
        assert model.config.input_dim == config["input_dim"]
        assert model.config.latent_dim == config["latent_dim"]
        assert model.config.hidden_dim == config["hidden_dim"]
