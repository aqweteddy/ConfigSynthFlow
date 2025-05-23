from config_synth_flow.base import BaseExecutor, BasePipeline, BaseReader, BaseWriter


class SequentialExecutor(BaseExecutor):
    def post_init(
        self,
        reader: BaseReader,
        writer: BaseWriter = None,
        pipes: list[BasePipeline] = None,
    ):
        """Initializes the SequentialExecutor.

        Args:
            reader (BaseReader): The reader instance.
            writer (BaseWriter, optional): The writer instance. Defaults to None.
            pipes (list[BasePipeline], optional): List of pipeline instances. Defaults to None.
        """
        super().post_init(reader=reader, writer=writer)
        self.pipes = pipes or []
        if self.writer is not None:
            pipes.append(self.writer)

    def execute(self) -> None:
        """Executes the pipeline sequentially."""
        dcts = self.reader()
        for pipe in self.pipes:
            dcts = pipe(dcts)
