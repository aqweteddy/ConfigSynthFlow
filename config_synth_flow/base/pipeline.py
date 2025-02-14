import asyncio
import importlib
import inspect
import logging
import os
import pickle
import pkgutil
import re
from collections.abc import Generator
from importlib import import_module, util
from itertools import islice
from logging import Formatter, StreamHandler, getLogger
from typing import Any, Callable, Union

import yaml
from tqdm.asyncio import tqdm as atqdm

from .config import PipelineConfig

DictsGenerator = Union[list[dict[str, Any]], Generator[dict[str, Any], None, None]]
"""
Data type for each pipeline input/output. It can be a list of dictionaries or a generator of dictionaries.
"""


def get_all_subclasses(package_name, base_class):
    subclasses = set()  # 使用集合來避免重複
    base_class_name = f"{base_class.__module__}.{base_class.__name__}"

    # 遍歷 package 下的所有 module
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


def _wrap_lambda(lambda_func: callable, *args, **kwargs) -> Callable:
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

    class WrappedLambda(BasePipeline):
        def run_each(self, dct: dict) -> dict:
            res = lambda_func(dct, *args, **kwargs)
            if isinstance(res, dict):
                for k, v in res.items():
                    dct[k] = v
                return dct
            elif isinstance(res, list):
                return res
            elif isinstance(res, bool):
                if res:
                    return dct
            else:
                raise ValueError("lambda function must return either a dict or a bool")

    cfg = PipelineConfig(init_kwargs={"init_kwargs": {}}, lambda_func=str(lambda_func))
    return WrappedLambda(cfg)


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
        ALL_PIPELINE_CLASSES = get_all_subclasses("config_synth_flow", BasePipeline)

    def get_import_path_by_class_name(import_path: str) -> BasePipeline:
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
        return _wrap_lambda(eval(config.lambda_func.strip()))
    else:
        raise ValueError(
            "config must have either `import_path` or `lambda_func` key. Got: "
            + str(config)
        )

    return getattr(m, func)(config)


class Serializable:
    def serialize(self):
        """
        Serialize the object to a byte stream.

        Returns:
            bytes: Serialized byte stream.
        """
        return pickle.dumps(self)

    @classmethod
    def deserialize(cls, data):
        """
        Deserialize the object from a byte stream.

        Args:
            data (bytes): Serialized byte stream.

        Returns:
            Serializable: Deserialized object.
        """
        return pickle.loads(data)

    def __getstate__(self):
        """
        Get the state of the object for serialization.

        Returns:
            dict: State of the object.
        """
        return self.__dict__

    def __setstate__(self, state):
        """
        Set the state of the object during deserialization.

        Args:
            state (dict): State of the object.
        """
        self.__dict__.update(state)


class EnvLoader(yaml.SafeLoader):
    pass


def env_var_constructor(loader, node):
    value = loader.construct_scalar(node)
    return os.getenv(value, f"Missing Env: {value}")


# Add the custom constructor for ${VAR_NAME}
EnvLoader.add_implicit_resolver("!env", re.compile(r"\$\{([^}]+)\}"), None)
EnvLoader.add_constructor("!env", env_var_constructor)


class BasePipeline(Serializable):
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
        self.class_name = self.__class__.__name__
        self.required_packages = getattr(self, "required_packages", [])
        self.check_required_packages()

        self.post_init(**config.init_kwargs)

    @property
    def logger(self):
        """
        Get the logger for the pipeline.

        Returns:
            logging.Logger: Logger for the pipeline.
        """
        if not hasattr(self, "_logger"):
            self._logger = getLogger(self.class_name)
            self._logger.setLevel(logging.INFO)
            handler = StreamHandler()
            formatter = Formatter(
                "%(asctime)s - %(name)s - %(levelname)s | %(message)s"
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
        return self._logger

    def check_required_packages(self) -> None:
        """
        Check if the required packages are installed.

        Raises:
            ImportError: If any of the required packages are missing.
        """

        missing_packages = [
            pkg for pkg in self.required_packages if not util.find_spec(pkg)
        ]
        if missing_packages:
            raise ImportError(
                f"Missing required packages: {missing_packages} in Pipeline {self.class_name}."
            )

    def post_init(self, **kwargs: Any) -> None:
        """
        Post initialization method. Write Pipeline initialization code here.

        Args:
            kwargs (Any): Initial kwargs.
        """
        ...

    def run_each(self, dct: dict) -> dict:
        """
        Run the pipeline on each input sample.

        Args:
            dct (dict): Input dictionary.
        Returns:
            dict: Output dictionary.
        """

        raise NotImplementedError(f"{self.class_name} is missing `run_each` method")

    def __call__(self, dcts: DictsGenerator) -> DictsGenerator:
        """
        Run the pipeline on a generator of input samples.

        Args:
            dcts (DictsGenerator): Input samples.
        Returns:
            DictsGenerator: Generator of output samples.
        """
        for dct in dcts:
            result = self.run_each(dct)
            if result:
                if isinstance(result, dict):
                    yield result
                else:
                    yield from result

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
                if isinstance(it, PipelineConfig):
                    pipes.append(cls.from_config(it))  # load_class
                else:
                    pipes.append(it)
            return pipes
        elif isinstance(config, dict):
            pipes = {}
            for k, v in config.items():
                if isinstance(v, PipelineConfig):
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


class AsyncBasePipeline(BasePipeline):
    def __init__(self, config: PipelineConfig):
        """
        Initialize the asynchronous pipeline.

        Args:
            config (PipelineConfig): Pipeline configuration.
        """
        super().__init__(config)
        config.async_cfg.batch_size = config.async_cfg.batch_size or 1000
        config.async_cfg.concurrency = config.async_cfg.concurrency or 100

    @property
    def semaphore(self):
        """
        Get the semaphore for limiting concurrency.

        Returns:
            asyncio.Semaphore: Semaphore for limiting concurrency.
        """
        return asyncio.Semaphore(self.config.async_cfg.concurrency)

    async def run_each(self, dct: dict) -> dict:
        """
        Run the pipeline on each input sample asynchronously.

        Args:
            dct (dict): Input dictionary.
        Returns:
            dict: Output dictionary.
        """

        raise NotImplementedError(f"{self.class_name} is missing `run_each` method")

    async def _async_call(self, dcts: list[dict]) -> list[dict]:
        """
        Run the pipeline on a list of input samples asynchronously.

        Args:
            dcts (list[dict]): List of input samples.

        Returns:
            list[dict]: List of output samples.
        """
        sem = asyncio.Semaphore(self.config.async_cfg.concurrency)

        async def limit_concurrency(sem, coro):
            async with sem:
                return await coro

        tasks = []
        for dct in dcts:
            tasks.append(limit_concurrency(sem, self.run_each(dct)))

        return await atqdm.gather(*tasks, desc=self.class_name)

    def get_chunk(self, dcts: DictsGenerator, chunk_size: int) -> DictsGenerator:
        """
        Get chunks of input samples.

        Args:
            dcts (DictsGenerator): Input samples.
            chunk_size (int): Size of each chunk.

        Returns:
            DictsGenerator: Generator of chunks of input samples.
        """
        it = iter(dcts)
        while chunk := list(islice(it, chunk_size)):
            yield list(chunk)

    def __call__(self, dcts: DictsGenerator) -> DictsGenerator:
        """
        Yield the output of the pipeline in chunks.

        Args:
            dcts (DictsGenerator): Input samples.
        Returns:
            DictsGenerator: Generator of output samples.
        """

        for chunk in self.get_chunk(dcts, self.config.async_cfg.batch_size):
            yield from asyncio.run(self._async_call(chunk))


ALL_PIPELINE_CLASSES = None
