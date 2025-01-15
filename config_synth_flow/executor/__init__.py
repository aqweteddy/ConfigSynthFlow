from ..base import BasePipeline, DictsGenerator
from datasets import Dataset, load_dataset
from tqdm import trange


class SeqentialExecutor(BasePipeline):
    def __post_init__(
        self,
        dataset_kwargs: dict,
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
        
        self.dataset_kwargs = dataset_kwargs
        self.pipes = pipes
        self.chunk_size = chunk_size

        self.ds = self.load_dataset(dataset_kwargs)
        self.data_num_proc = dataset_kwargs.get("num_proc", 4)
        self.output_path = output_path
        self.resume = resume

    def load_dataset(self, dataset_kwargs: dict) -> Dataset:
        ds = load_dataset(**dataset_kwargs)
        if not isinstance(ds, Dataset):
            ds = ds["train"]

        self.logger.info(f"Dataset kwargs: {dataset_kwargs}")
        self.logger.info(f"Numbers of samples: {len(ds)}")
        self.logger.info(f"Columns: {ds.column_names}")

        if self.resume:
            self.logger.info("Resuming previous run")
            processed_ds = load_dataset(
                "json", data_files=self.output_path, num_proc=self.data_num_proc
            )
            self.processed_ids = set(processed_ds["hash_id"])
        return ds

    @staticmethod
    def map_hash_id(dct: dict) -> dict:
        dct["hash_id"] = str(hash(dct))[:8]
        return dct

    def chunked_run(self, chunk_size: int = None) -> None:
        """
        Chunked the dataset and run the pipelines sequentially.
        
        Args:
            chunk_size (int, optional): Chunk size to split the dataset. Defaults to None.
        """
        
        chunk_size = chunk_size or self.chunk_size
        num_shards = len(self.ds) // chunk_size
        self.logger.info(f"Splitting dataset into {num_shards} chunks.")

        self.config.save_yaml(self.output_path + "/config.yml")

        for i in trange(num_shards, desc="Chunked Run"):
            sub_ds = self.ds.shard(num_shards, i, contiguous=True)

            # For resuming, we need to get the hash_id of each sample
            sub_ds = sub_ds.map(self.map_hash_id, num_proc=self.data_num_proc)

            # Skip if the chunk is already processed when resuming
            if self.resume:
                sub_ds = sub_ds.filter(
                    lambda x: x["hash_id"] not in self.processed_ids,
                    num_proc=self.data_num_proc,
                )
                if len(sub_ds) == 0:
                    self.logger.info(f"Chunk {i} already processed. Skipping.")
                    continue

            # Run the pipelines
            result_ds = self(sub_ds)

            output_file = f"{self.output_path}/chunk_{i:3d}.json"
            self.logger.info(f"Saving chunk {i} to {output_file}")
            result_ds.to_json(output_file, force_ascii=False)
            
            sub_ds.cleanup_cache_files()
            result_ds.cleanup_cache_files()

    def __call__(self, dcts: DictsGenerator) -> Dataset:
        for pipe in self.pipes:
            dcts = pipe(dcts)

        return Dataset.from_list(list(dcts))
