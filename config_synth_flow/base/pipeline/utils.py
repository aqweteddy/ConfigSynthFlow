"""
Utility functions for pipeline module.
"""

import importlib
import inspect
import os
import pkgutil
import re
from importlib import import_module
from typing import Callable

import yaml

from .config import PipelineConfig

# This will be set by the base module
ALL_PIPELINE_CLASSES = None


def get_all_subclasses(package_name, base_class):
    """
    Get all subclasses of a base class in a package.

    Args:
        package_name (str): Name of the package to search in.
        base_class (type): Base class to find subclasses of.

    Returns:
        set: Set of fully qualified class names.
    """
    subclasses = set()
    base_class_name = f"{base_class.__module__}.{base_class.__name__}"

    for _, module_name, is_pkg in pkgutil.walk_packages(
        importlib.import_module(package_name).__path__, package_name + "."
    ):
        if not is_pkg:
            try:
                module = importlib.import_module(module_name)
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(obj, base_class)
                        and f"{obj.__module__}.{obj.__name__}" != base_class_name
                    ):
                        subclasses.add(f"{obj.__module__}.{obj.__name__}")
            except ImportError:
                continue
    return subclasses


def _wrap_lambda(config: PipelineConfig, *args, **kwargs) -> Callable:
    """
    Wrap a lambda function to be used as a pipeline.
    There are two ways to use the lambda function:
    - If the lambda function returns a dict, update the input dict with the returned dict.
    - If the lambda function returns a bool, yield the input dict if the returned value is True.

    Args:
        lambda_func (callable): Lambda function to be wrapped.
        args: Arguments for the lambda function.
        kwargs: Keyword arguments for the lambda function.
    Returns:
        Callable: Wrapped lambda function.
    """
    # Import here to avoid circular imports
    from .lambda_pipeline import LambdaFunc

    config.init_kwargs = {
        "lambda_func": config.lambda_func,
    }
    config.lambda_func = None

    return LambdaFunc(config)


def _get_class(config: PipelineConfig):
    """
    Get and instantiate a pipeline class according to the config.import_path or config.lambda_func.

    Args:
        config (PipelineConfig): Pipeline configuration.

    Returns:
        Any: Instantiated pipeline class or wrapped lambda function.
    """
    global ALL_PIPELINE_CLASSES
    if ALL_PIPELINE_CLASSES is None:
        # Import here to avoid circular imports
        from .base_pipeline import BasePipeline

        ALL_PIPELINE_CLASSES = get_all_subclasses("config_synth_flow", BasePipeline)

    def get_import_path_by_class_name(import_path: str) -> str:
        """
        Find the absolute path of the pipeline class.

        Args:
            import_path (str): Import path to the pipeline class.

        Returns:
            str: Absolute path of the pipeline class.
        """
        matched = []
        for cls in ALL_PIPELINE_CLASSES:
            if import_path.strip() == cls.split(".")[-1].strip():
                matched.append(cls)

        if len(matched) == 1:
            return matched[0]
        elif len(matched) == 0:
            return import_path
        else:
            raise ValueError(
                f"Multiple classes found for {import_path}. Please provide the full path. The matched classes are: {matched}"
            )

    if config.import_path:
        config.import_path = get_import_path_by_class_name(config.import_path)
        try:
            module_, func = config.import_path.rsplit(".", maxsplit=1)
            m = import_module(module_)
        except (ImportError, ModuleNotFoundError, ValueError):
            raise ImportError(
                f"Cannot import module: {config.import_path}. Check whether the import_path is correct."
            )
    elif config.lambda_func:
        return _wrap_lambda(config)
    else:
        raise ValueError(
            "config must have either `import_path` or `lambda_func` key. Got: " + str(config)
        )

    return getattr(m, func)(config)


class EnvLoader(yaml.SafeLoader):
    """Custom YAML loader that supports environment variables."""

    pass


def env_var_constructor(loader, node):
    """Constructor for environment variables in YAML."""
    value = loader.construct_scalar(node)
    return os.getenv(value, f"Missing Env: {value}")


# Add the custom constructor for ${VAR_NAME}
EnvLoader.add_implicit_resolver("!env", re.compile(r"\$\{([^}]+)\}"), None)
EnvLoader.add_constructor("!env", env_var_constructor)
