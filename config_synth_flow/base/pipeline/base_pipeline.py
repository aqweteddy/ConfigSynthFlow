"""
Base pipeline implementation.
"""

import yaml

from ..mixins import (
    LoggingMixin,
    PipelineCoreMixin,
    RequiredPackagesMixin,
    SerializableMixin,
)
from .config import PipelineConfig
from .utils import EnvLoader


class BasePipeline(SerializableMixin, LoggingMixin, RequiredPackagesMixin, PipelineCoreMixin):
    """Base class for all pipelines."""

    class_name: str
    config: PipelineConfig
    required_packages: list[str]

    def __init__(self, config: PipelineConfig) -> None:
        """
        Base class for all pipelines.

        Args:
            config (PipelineConfig): Pipeline configuration.
        """
        self.config = config
        self.required_packages = getattr(self, "required_packages", [])
        self.check_required_packages()

        self.post_init(**config.init_kwargs)

    def __repr__(self):
        """
        Get the string representation of the pipeline.

        Returns:
            str: String representation of the pipeline.
        """
        repr = f"{self.class_name}\n"
        repr += " " * 4 + str(self.config)
        return repr

    @classmethod
    def from_config(cls, config: list | dict | PipelineConfig) -> "BasePipeline":
        """
        Recursive create Pipeline from a configuration.
        Notice that the top-level config must be a PipelineConfig.

        Args:
            config (list | dict | PipelineConfig): Configuration of the pipeline.
        Returns:
            BasePipeline: Pipeline object.
        """
        # Import here to avoid circular imports
        from .utils import _get_class

        if isinstance(config, PipelineConfig):  # load `BasePipeline` class
            for k, v in config.init_kwargs.items():
                if isinstance(v, (PipelineConfig, list, dict)):
                    config.init_kwargs[k] = cls.from_config(v)
            return _get_class(config)
        elif isinstance(
            config, list
        ):  # if config is a list, iterate over the list and call `from_config` recursively
            pipes = []
            for it in config:
                if isinstance(it, (PipelineConfig, list, dict)):
                    pipes.append(cls.from_config(it))  # load_class
                else:
                    pipes.append(it)
            return pipes
        elif isinstance(config, dict):
            pipes = {}
            for k, v in config.items():
                if isinstance(v, (PipelineConfig, list, dict)):
                    pipes[k] = cls.from_config(v)
                else:
                    pipes[k] = v
            return pipes
        else:
            raise ValueError("config must be either a list, a dict or a PipelineConfig")

    @classmethod
    def from_yaml(cls, path: str) -> "BasePipeline":
        """
        Load the pipeline from a yaml file.

        Args:
            path (str): Path to the yaml file.
        Returns:
            BasePipeline: Pipeline object.
        """
        with open(path) as f:
            cfg = yaml.load(f, Loader=EnvLoader)
        cfg = PipelineConfig(**cfg)
        return cls.from_config(cfg)
