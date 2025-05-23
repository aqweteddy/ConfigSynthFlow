from datasets import Dataset, IterableDataset, load_dataset

from config_synth_flow.base import BaseReader, DictsGenerator


class HfDatasetReader(BaseReader):
    required_packages: list = ["datasets"]

    def post_init(
        self,
        dataset_kwargs: dict,
        resume: bool = False,
        debug: bool = False,
        shuffle: bool = False,
    ):
        """
        Read the dataset from the given kwargs.

        Args:
            dataset_kwargs (dict): Dataset kwargs to load the dataset. refer to `datasets.load_dataset` for more details.
            resume (bool): Whether to resume the reading from the last processed sample.
            debug (bool): Whether to debug the reading process.
            shuffle (bool): Whether to shuffle the dataset.
        """
        super().post_init(resume=resume)

        self.dataset_kwargs = dataset_kwargs
        self.ds = self.load_dataset(dataset_kwargs)
        self.num_proc = self.dataset_kwargs.get("num_proc", 4)
        if shuffle:
            self.ds = self.ds.shuffle()

        if debug:
            num = 10 if debug is True else debug
            if isinstance(self.ds, IterableDataset):
                self.ds = self.ds.take(num)
            else:
                self.ds = self.ds.select(range(num))

    def load_dataset(self, dataset_kwargs: dict) -> Dataset:
        ds = load_dataset(**dataset_kwargs)
        if not isinstance(ds, Dataset):
            ds = ds["train"]

        self.logger.info(f"Dataset kwargs: {dataset_kwargs}")
        if not isinstance(ds, IterableDataset):
            self.logger.info(f"Numbers of samples: {len(ds)}")
            self.logger.info(f"Columns: {ds.column_names}")
        else:
            self.logger.info("Numbers of samples: Unknown")

        return ds

    def read(self) -> DictsGenerator:
        cnt = 0
        for dct in self.ds:
            cnt += 1
            if isinstance(self.ds, IterableDataset) and cnt % 1000 == 0:
                self.logger.info(f"read {cnt} samples")
            yield dct
