import itertools
from concurrent.futures import ProcessPoolExecutor

from config_synth_flow.base import BasePipeline, DictsGenerator, PipelineConfig
from config_synth_flow.reader import BaseReader
from config_synth_flow.writer import BaseWriter

from .base import BaseExecutor


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


class PipelineRunner(BasePipeline):
    def __init__(self, pipe_cfg_list: list[PipelineConfig]):
        """Initializes the PipelineRunner.

        Args:
            pipe_cfg_list (list[PipelineConfig]): List of pipeline configurations.
        """
        self.pipes = [BasePipeline.from_config(pipe_cfg) for pipe_cfg in pipe_cfg_list]

    def __call__(self, dataset: DictsGenerator) -> DictsGenerator:
        """Runs the dataset through the pipeline.

        Args:
            dataset (DictsGenerator): The dataset generator.

        Returns:
            DictsGenerator: The processed dataset.
        """
        for pipe in self.pipes:
            dataset = pipe(dataset)

        return list(dataset)


class MultiProcessSequentialExecutor(BaseExecutor):
    def post_init(
        self,
        reader: BaseReader,
        writer: BaseWriter = None,
        pipes: list[BasePipeline] = None,
        num_proc: int = 4,
        chunk_size: int = 8,
    ):
        """Initializes the MultiProcessSequentialExecutor.

        Args:
            reader (BaseReader): The reader instance.
            writer (BaseWriter, optional): The writer instance. Defaults to None.
            pipes (list[BasePipeline], optional): List of pipeline instances. Defaults to None.
            num_proc (int, optional): Number of processes. Defaults to 4.
            chunk_size (int, optional): Size of each chunk. Defaults to 8.
        """
        super().post_init(reader=reader, writer=writer)
        self.pipes = pipes or []
        self.num_proc = num_proc
        self.chunk_size = chunk_size
        if self.writer is not None:
            pipes = pipes[:-1]

    def yield_dcts_chunk(self, dcts: DictsGenerator, chunk_size: int):
        """Yields chunks of dictionaries.

        Args:
            dcts (DictsGenerator): The dataset generator.
            chunk_size (int): Size of each chunk.

        Yields:
            list: A chunk of dictionaries.
        """
        it = iter(dcts)
        while chunk := list(itertools.islice(it, chunk_size)):
            yield list(chunk)

    def execute(self) -> None:
        """Executes the pipeline using multiple processes."""
        dcts = self.reader()
        pipe_runner = PipelineRunner(pipe_cfg_list=[pipe.config for pipe in self.pipes])
        with ProcessPoolExecutor(max_workers=self.num_proc) as executor:
            res_list = list(
                executor.map(pipe_runner, self.yield_dcts_chunk(dcts, self.chunk_size))
            )

        for res in res_list:
            self.writer(res)
