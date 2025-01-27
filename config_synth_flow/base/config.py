from pathlib import Path
from typing import Any, Callable, ForwardRef

import yaml
from pydantic import BaseModel

PipelineConfig = ForwardRef("PipelineConfig")


class AsyncConfig(BaseModel):
    """
    Configuration for an asynchronous pipeline.

    Attributes:
        async_concurrency (int): The number of concurrent tasks to run.
        async_batch_size (int): The number of tasks to run in each batch.
    """

    concurrency: int | None = None
    batch_size: int | None = None


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
        async_config (AsyncConfig): Configuration for an asynchronous pipeline.
    """

    import_path: str | None = None
    lambda_func: str | Callable | None = None
    cfg_path: str | None = None
    init_kwargs: (
        dict[
            str, PipelineConfig | list[PipelineConfig] | dict[str, PipelineConfig] | Any
        ]
        | None
    ) = None

    # async only
    async_cfg: AsyncConfig | None = None

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
        
        if (
            self.import_path is None
            and self.lambda_func is None
            and self.cfg_path is None
        ):
            raise ValueError(
                "config must have either `import_path` or `lambda_func` key or `cfg_path` key. Got: "
                + str(self)
            )

        self.init_kwargs = self.init_kwargs or {}

        if self.cfg_path is not None:
            with yaml.unsafe_load(self.cfg_path, "r") as f:
                cfg = f.read()
            if isinstance(cfg, dict):
                self.init_kwargs = cfg["init_kwargs"]
                self.import_path = cfg["import_path"]
            else:
                raise ValueError(
                    f"Invalid configuration file: {self.cfg_path}. Notice that the file must be a dictionary."
                )

        return self

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
