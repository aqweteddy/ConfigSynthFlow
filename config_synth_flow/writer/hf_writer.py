from typing import Literal

from datasets import Dataset

from config_synth_flow.base import BaseWriter, DictsGenerator


class HfWriter(BaseWriter):
    def post_init(
        self,
        output_path: str,
        chunk_size: int = 1000,
        output_format: Literal["jsonl", "json", "csv", "parquet"] = "jsonl",
    ):
        """
        Initialize the HfWriter.

        Args:
            output_path (str): Path to the output directory.
            chunk_size (int): Number of records per chunk. Defaults to 1000.
            output_format (Literal["jsonl", "json", "csv", "parquet"]): Output format. Defaults to "jsonl".
        """
        super().post_init(output_path=output_path)
        self.output_format = output_format
        self.chunk_size = chunk_size
        self.chunk_id = 0

    def save_hf_dataset(self, dcts: list[dict]) -> None:
        """
        Save a list of dictionaries as a Hugging Face dataset.

        Args:
            dcts (list[dict]): List of dictionaries to save.
        """
        ds = Dataset.from_list(dcts)
        while True:
            output_path = self.output_path / f"chunk_{self.chunk_id:05d}.{self.output_format}"
            if not output_path.exists():
                break
            self.chunk_id += 1
        self.logger.info(f"Saving result to {output_path}")
        if self.output_format == "jsonl":
            ds.to_json(output_path, force_ascii=False)
        elif self.output_format == "json":
            ds.to_json(output_path, force_ascii=False)
        elif self.output_format == "csv":
            ds.to_csv(output_path)
        elif self.output_format == "parquet":
            ds.to_parquet(output_path)
        else:
            raise ValueError(f"Unsupported output format: {self.output_format}")
        ds.cleanup_cache_files()

    def __call__(self, dataset: DictsGenerator) -> None:
        """
        Write the result to the output path.

        Args:
            dataset (DictsGenerator): Dataset to write.
        """
        output_list = []

        for dct in dataset:
            output_list.append(dct)
            if len(output_list) == self.chunk_size:
                self.save_hf_dataset(output_list)
                self.chunk_id += 1
                output_list = []

        if len(output_list) > 0:
            self.save_hf_dataset(output_list)
