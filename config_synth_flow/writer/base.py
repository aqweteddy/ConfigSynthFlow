from pathlib import Path

from ..base import BasePipeline


class BaseWriter(BasePipeline):
    def post_init(self, output_path: str = None):
        """
        Initialize the BaseWriter.

        Args:
            output_path (str): Path to the output directory.
        """
        self.output_path = Path(output_path)
