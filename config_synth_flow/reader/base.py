from typing import Any
from hashlib import md5
from datasets import load_dataset
from ..base import BasePipeline, DictsGenerator


class BaseReader(BasePipeline):
    """
    Base class for the reader.
    Read the dataset and generate the dict[str, Any] for the pipeline.
    override the `read` method to read the dataset. You can also override the `get_unique_id` method to generate the unique id for each sample.
    Notice that __call__ method is implemented to generate the unique id for each sample.
    
    property:
    - resume: bool: Resume the previous run. Default to False. When True, it will load the processed dataset and skip the processed samples.
    """
    
    resume: bool = False

    def __call__(self) -> DictsGenerator:
        """
        implicit to call the `read` method, generate the unique id for each sample and yield the sample.
        """
        for dct in self.read():
            dct['hash_id'] = self.get_unique_id(dct)
            if self.resume and dct['hash_id'] in self.unique_id_set:
                continue
            yield dct
        
    def read(self) -> DictsGenerator:
        """
        Implement the read method to read the dataset.
        """
        ...

    @staticmethod
    def get_unique_id(obj: Any) -> str:
        return md5(str(obj).encode()).hexdigest()

    def set_done_ids(self, output_path: str, num_proc: int = 4) -> None:
        """
        Set the done ids from the processed dataset.
        
        Args:
            output_path (str): Output path to save the processed dataset.
            num_proc (int, optional): Number of processes to load the dataset. Defaults to 4.
        
        """
        ds = load_dataset("json", data_dir=output_path, num_proc=num_proc)["train"]
        self.unique_id_set = set(ds["hash_id"])
        self.resume = True
        ds.cleanup_cache_files()
