from ..base import DictsGenerator
from datasets import load_dataset, Dataset, IterableDataset
from .base import BaseReader


class HfDatasetReader(BaseReader):
    required_packages: list = ["datasets"]

    def __post_init__(self, resume: bool, dataset_kwargs: dict, debug: bool = False):
        """
        Read the dataset from the given kwargs.

        Args:
            dataset_kwargs (dict): Dataset kwargs to load the dataset. refer to `datasets.load_dataset` for more details.
        """
        super().__post_init__(resume=resume)

        self.dataset_kwargs = dataset_kwargs
        self.ds = self.load_dataset(dataset_kwargs)
        self.num_proc = self.dataset_kwargs.get("num_proc", 4)

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
