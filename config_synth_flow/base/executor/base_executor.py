from pprint import pformat

from ..io import BaseReader, BaseWriter
from ..pipeline import BasePipeline, PipelineConfig


class BaseExecutor(BasePipeline):
    def __init__(self, config: PipelineConfig):
        """Initializes the BaseExecutor.

        Args:
            config (PipelineConfig): The pipeline configuration.
        """
        super().__init__(config)
        self.reader.set_writer_output_path(self.writer.output_path)
        if self.writer is not None and self.writer.output_path is not None:
            self.config.save(self.writer.output_path / "config.yml")
        self.logger.info(f"Executor config:\n{pformat(self.config, indent=2)}")

    def post_init(
        self,
        reader: BaseReader,
        writer: BaseWriter,
    ):
        """Post-initializes the BaseExecutor. Call by __init__ of child classes.
        In this method, you can set your custom `init_kwargs` for the executor.
        For Executor, it should at least set the reader and writer instances.

        Args:
            reader (BaseReader): The reader instance.
            writer (BaseWriter): The writer instance.
        """
        self.reader = reader
        self.writer = writer

    def execute(self) -> None:
        """Executes the pipeline from reader to writer."""
        raise NotImplementedError
