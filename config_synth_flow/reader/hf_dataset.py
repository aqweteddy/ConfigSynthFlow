from ..base import DictsGenerator
from datasets import load_dataset, Dataset
from .base import BaseReader


class HfDatasetReader(BaseReader):
    def __post_init__(self, dataset_kwargs: dict):
        """
        Read the dataset from the given kwargs.
        
        Args:
            dataset_kwargs (dict): Dataset kwargs to load the dataset. refer to `datasets.load_dataset` for more details.
        """
        self.dataset_kwargs = dataset_kwargs
        self.ds = self.load_dataset(dataset_kwargs)
        self.num_proc = self.dataset_kwargs.get("num_proc", 4)

    def load_dataset(self, dataset_kwargs: dict) -> Dataset:
        ds = load_dataset(**dataset_kwargs)
        if not isinstance(ds, Dataset):
            ds = ds["train"]

        self.logger.info(f"Dataset kwargs: {dataset_kwargs}")
        self.logger.info(f"Numbers of samples: {len(ds)}")
        self.logger.info(f"Columns: {ds.column_names}")

        return ds
    
    def read(self) -> DictsGenerator:
        for dct in self.ds:
            yield dct
