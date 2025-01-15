from ..base import BasePipeline, DictsGenerator
from ..reader import BaseReader
from datasets import Dataset
from pathlib import Path


class SequentialExecutor(BasePipeline):
    def __post_init__(
        self,
        reader: BaseReader,
        output_path: str = None,
        chunk_size: int = 1000,
        resume: bool = False,
        pipes: list[BasePipeline] = None,
    ):
        """
        Sequentially run pipelines on the dataset.

        Args:
            dataset_kwargs (dict): Dataset kwargs to load the dataset. refer to `datasets.load_dataset` for more details.
            output_path (str, optional): Output path to save the processed dataset. Defaults to None.
            chunk_size (int, optional): Chunk size to split the dataset. Defaults to 1000.
            resume (bool, optional): Resume the previous run. Defaults to False.
            pipes (list[BasePipeline], optional): List of pipelines to run sequentially.
        """
        self.reader = reader
        self.pipes = pipes
        self.chunk_size = chunk_size
        self.output_path = Path(output_path)
        self.resume = resume
        
        self.logger.info(f"Pipeline overview:\n{self}")

    def get_start_id(
        self,
    ) -> int:
        if self.resume:
            done_files = self.output_path.glob("*.jsonl")
            if len(done_files) == 0:
                return 0
            return max([int(f.stem.split("_")[-1]) for f in done_files])
        return 0

    def chunked_run(self, chunk_size: int = None) -> None:
        """
        Chunked the dataset and run the pipelines sequentially.

        Args:
            chunk_size (int, optional): Chunk size to split the dataset. Defaults to None.
        """

        self.config.save(self.output_path / "config.yml")
        if self.resume:
            self.reader.set_done_ids(self.output_path)

        dcts = []
        chunk_id = self.get_start_id()

        for dct in self.reader():
            dcts.append(dct)
            if len(dcts) == chunk_size:
                self.logger.info(f"Processing chunk {chunk_id}")
                result_ds = self(dcts)
                dcts = []
                self.write_output(result_ds, self.output_path / f"{chunk_id:05d}.jsonl")
                chunk_id += 1

        if len(dcts) > 0:
            result_ds = self(dcts)
            self.write_output(result_ds, self.output_path / f"{chunk_id:05d}.jsonl")

        self.logger.info("All Done!")

    def write_output(self, result_ds: Dataset, output_path: str) -> None:
        """
        Write the processed dataset to the output path.

        Args:
            result_ds (Dataset): Processed dataset.
        """
        self.logger.info(f"Writing to {output_path}")
        result_ds.to_json(
            output_path,
            force_ascii=False,
        )

    def __call__(self, dcts: DictsGenerator) -> Dataset:
        for pipe in self.pipes:
            dcts = pipe(dcts)

        return Dataset.from_list(list(dcts))
