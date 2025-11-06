"""Global test configuration."""
import pytest
from pathlib import Path
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict
import rootutils


@pytest.fixture(scope="package")
def cfg_train_global() -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for training."""
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()
    
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml", return_hydra_config=True, overrides=[])

        # Set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_train_batches = 0.01
            cfg.trainer.limit_val_batches = 0.1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            
            # Create extras if it doesn't exist
            if "extras" not in cfg:
                cfg.extras = {}
            cfg.extras.print_config = False
            cfg.extras.ignore_warnings = True

    yield cfg
    
    # Cleanup
    GlobalHydra.instance().clear()


@pytest.fixture(scope="package")
def cfg_eval_global() -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for evaluation."""
    GlobalHydra.instance().clear()
    
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="eval.yaml", return_hydra_config=True, overrides=[])

        with open_dict(cfg):
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            
            if "extras" not in cfg:
                cfg.extras = {}
            cfg.extras.print_config = False

    yield cfg
    
    GlobalHydra.instance().clear()


# Aliases for backward compatibility
@pytest.fixture(scope="package")
def cfg_train(cfg_train_global: DictConfig) -> DictConfig:
    """Alias for cfg_train_global."""
    return cfg_train_global


@pytest.fixture(scope="package")
def cfg_eval(cfg_eval_global: DictConfig) -> DictConfig:
    """Alias for cfg_eval_global."""
    return cfg_eval_global