from ..base import BasePipeline
from pathlib import Path


class BaseWriter(BasePipeline):
    def __post_init__(self, 
                      output_path: str = None
    ):
        self.output_path = Path(output_path)
    