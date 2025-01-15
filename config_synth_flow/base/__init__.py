import yaml

from pydantic import BaseModel
from typing import Any, Generator, ForwardRef, Union, Callable
from importlib import import_module
from abc import ABC
from logging import getLogger, Formatter, StreamHandler
import logging


DictsGenerator = Union[list[dict[str, Any]], Generator[dict[str, Any], None, None]]

PipelineConfig = ForwardRef("PipelineConfig")


class PipelineConfig(BaseModel):
    """
    Configuration for a pipeline.
    You have two ways to create a pipeline:
    - import_path: Will import the class from the import_path.
    - lambda_func: Will create a lambda function to be used as a pipeline.

    And you have two ways to provide the initial kwargs of Pipeline class:
    - init_kwargs: Initial kwargs for the pipeline class.
    - cfg_path: Path to the yaml file containing the configuration of the pipeline.

    For example:
    - Use import_path to import the class from the import_path.

    ```python
    PipelineConfig(
        import_path="agent_chat_gen.pipelines.api.openai_chat.BatchOpenAIChat",
        init_kwargs={
            "model": "gpt-4o-mini",
            "openai_kwargs": None,
            "gen_kwargs": None,
            "messages_col": "messages",
            "output_col": "_response",
        },
    )
    ```

    - Use lambda_func to create a lambda function to be used as a pipeline.

    ```python
    PipelineConfig(
        lambda_func="lambda x: x",
    )
    ```

    - Use cfg_path to load the configuration from a yaml file.

    ```python

    PipelineConfig(
        cfg_path="config.yml",
    )
    ```

    Attributes:
        import_path (str): Import path to the pipeline class.
        lambda_func (str | Callable): Lambda function to be used as a pipeline.
        init_kwargs (dict[str, PipelineConfig | Any]): Initial kwargs for the pipeline class.
        cfg_path (str): Path to the yaml file containing the configuration of the pipeline.
    """

    import_path: str | None = None
    lambda_func: str | Callable | None = None
    init_kwargs: dict[str, PipelineConfig | Any] | None = None
    cfg_path: str | None = None

    def model_post_init(self, _):
        if (
            self.import_path is None
            and self.lambda_func is None
            or self.cfg_path is None
        ):
            raise ValueError(
                "config must have either `import_path` or `lambda_func` key or `cfg_path` key. Got: "
                + str(self)
            )

        if self.lambda_func is not None:
            self.init_kwargs = self.init_kwargs or {}

        if self.cfg_path is not None:
            with yaml.unsafe_load(self.cfg_path, "r") as f:
                cfg = f.read()
            self.init_kwargs = cfg["init_kwargs"]
            self.import_path = cfg["import_path"]

        return self

    def save_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f)


PipelineConfig.model_rebuild()


def wrap_lambda(lambda_func: callable, *args, **kwargs) -> Callable:
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

    def __wrapped(dcts: DictsGenerator) -> DictsGenerator:  # type: ignore
        for dct in dcts:
            try:
                res = lambda_func(dct, *args, **kwargs)
            except TypeError as e:
                print(f"Error in {lambda_func.__name__}: {e}")
                raise e
            if isinstance(res, dict):
                for k, v in res.items():
                    dct[k] = v
                yield dct
            elif isinstance(res, bool):
                if res:
                    yield dct
            else:
                raise ValueError("lambda function must return either a dict or a bool")

    return __wrapped


class BasePipeline(ABC):
    class_name: str
    config: PipelineConfig

    def __init__(self, config: PipelineConfig) -> None:
        """
        Base class for all pipelines.

        Args:
            config (PipelineConfig): Pipeline configuration.
        """

        self.config = config
        self.class_name = self.__class__.__name__
        self.__post_init__(**config.init_kwargs)

    @property
    def logger(self):
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

    def __post_init__(self, **kwargs: Any) -> None:
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
                yield result

    @staticmethod
    def get_class(config: PipelineConfig):
        """
        get and instantiate a pipeline class according to the config.import_path or config.lambda_func
        """
        if config.import_path:
            module_, func = config.import_path.rsplit(".", maxsplit=1)
            m = import_module(module_)
        elif config.lambda_func:
            return wrap_lambda(eval(config.lambda_func))
        else:
            raise ValueError(
                "config must have either `import_path` or `lambda_func` key. Got: "
                + str(config)
            )

        return getattr(m, func)(config)

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
        if isinstance(config, PipelineConfig):
            for k, v in config.init_kwargs.items():
                if isinstance(v, PipelineConfig):
                    config.init_kwargs[k] = cls.from_config(v)
                else:
                    config.init_kwargs[k] = v
            return cls.get_class(config)
        elif isinstance(config, list):
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
            raise ValueError(
                "config must be either a list or a dict or a PipelineConfig"
            )
