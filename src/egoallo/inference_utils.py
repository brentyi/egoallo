"""Functions that are useful for inference scripts."""

from pathlib import Path

import yaml
from safetensors import safe_open

from .network import EgoDenoiser, EgoDenoiserConfig


def load_denoiser(checkpoint_dir: Path) -> EgoDenoiser:
    """Load a denoiser model."""
    checkpoint_dir = checkpoint_dir.absolute()
    experiment_dir = checkpoint_dir.parent

    config = yaml.load(
        (experiment_dir / "model_config.yaml").read_text(), Loader=yaml.Loader
    )
    assert isinstance(config, EgoDenoiserConfig)

    model = EgoDenoiser(config)
    with safe_open(checkpoint_dir / "model.safetensors", framework="pt") as f:  # type: ignore
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    model.load_state_dict(state_dict)

    return model
