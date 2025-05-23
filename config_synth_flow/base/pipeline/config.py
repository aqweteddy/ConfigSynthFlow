from pathlib import Path
from typing import Any, ForwardRef

import yaml
from pydantic import BaseModel

from ..prompt_template import PromptTemplate

PipelineConfig = ForwardRef("PipelineConfig")


class AsyncConfig(BaseModel):
    """
    Configuration for an asynchronous pipeline.

    Attributes:
        concurrency (int): The number of concurrent tasks to run.
        batch_size (int): The number of tasks to run in each batch.
    """

    concurrency: int | None = None
    batch_size: int | None = None
    show_progress: bool | None = True


class MultiProcessConfig(BaseModel):
    """
    Configuration for a multi-process pipeline.

    Attributes:
        num_proc (int): The number of processes to use.
        chunk_size (int): The size of chunks to process in each process.
        max_queue_size (int): The maximum size of the queue for inter-process communication.
    """

    num_proc: int | None = None
    show_progress: bool | None = None


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
        async_cfg (AsyncConfig): Configuration for an asynchronous pipeline.
        mp_cfg (MultiProcessConfig): Configuration for a multi-process pipeline.
    """

    import_path: str | None = None
    lambda_func: str | None = None
    cfg_path: str | None = None
    init_kwargs: (
        dict[
            str,
            PipelineConfig
            | list[PipelineConfig]
            | dict[str, PipelineConfig]
            | PromptTemplate
            | list[PromptTemplate]
            | dict[str, PromptTemplate]
            # | list[dict[str, Any]]
            # | dict[str, Any]
            | Any,
        ]
        | None
    ) = None

    # async only
    async_cfg: AsyncConfig | None = None
    # multi-process only
    mp_cfg: MultiProcessConfig | None = None

    # private
    __original_kwargs: dict[str, Any]

    def model_post_init(self, _):
        """
        Post initialization method for the model.

        Args:
            _ (Any): Placeholder argument.
        """
        self.__original_kwargs = self.model_dump()
        self.async_cfg = self.async_cfg or AsyncConfig()
        self.mp_cfg = self.mp_cfg or MultiProcessConfig()

        if self.import_path is None and self.lambda_func is None and self.cfg_path is None:
            raise ValueError(
                "config must have either `import_path` or `lambda_func` key or `cfg_path` key. Got: "
                + str(self)
            )

        self.init_kwargs = self.init_kwargs or {}

        if self.cfg_path is not None:
            with open(self.cfg_path) as f:
                cfg = yaml.unsafe_load(f)

            if isinstance(cfg, dict):
                self.init_kwargs = cfg["init_kwargs"]
                self.import_path = cfg["import_path"]
            else:
                raise ValueError(
                    f"Invalid configuration file: {self.cfg_path}. Notice that the file must be a dictionary."
                )

        # Recursively convert nested dictionaries to PipelineConfig objects if they have import_path
        if self.init_kwargs:
            self.init_kwargs = self._process_nested_configs(self.init_kwargs)

    def _process_nested_configs(self, value):
        """
        Recursively process nested dictionaries and convert them to PipelineConfig objects if applicable.

        Args:
            value: The value to process.

        Returns:
            The processed value.
        """
        if isinstance(value, dict):
            # Check if this dictionary is a potential PipelineConfig
            if "import_path" in value or "lambda_func" in value or "cfg_path" in value:
                # If it's not already a PipelineConfig, convert it
                try:
                    return PipelineConfig(**value)
                except Exception as e:
                    print(e)
                    pass
                return value
            # Otherwise process each item in the dictionary
            return {k: self._process_nested_configs(v) for k, v in value.items()}
        elif isinstance(value, list):
            # Process each item in the list
            return [self._process_nested_configs(item) for item in value]
        elif isinstance(value, PipelineConfig):
            for k in value.init_kwargs:
                value.init_kwargs[k] = self._process_nested_configs(value.init_kwargs[k])
            return value
        else:
            return value

    def save(self, path: Path) -> None:
        """
        Save the pipeline configuration to a file.

        Args:
            path (Path): Path to the file where the configuration will be saved.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.__original_kwargs, f, allow_unicode=True)


PipelineConfig.model_rebuild()
