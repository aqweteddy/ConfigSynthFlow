from concurrent.futures import ProcessPoolExecutor

import ftfy
from trafilatura import extract

from ...base import BasePipeline, DictsGenerator


class TrafilaturaPipeline(BasePipeline):
    """Pipeline for extracting and cleaning text from HTML using Trafilatura."""

    required_packages = ["trafilatura"]

    def post_init(
        self,
        html_col: str = "text",
        output_col: str = "text",
        num_proc: int = 10,
    ):
        """
        Initialize the pipeline with column names and number of processes.

        Args:
            html_col (str): Column name containing HTML text.
            output_col (str): Column name to store the extracted text.
            num_proc (int): Number of processes to use for parallel processing.
        """
        self.html_col = html_col
        self.output_col = output_col
        self.num_proc = num_proc

    def run_each(self, dct: dict) -> dict:
        """
        Extract and clean text from HTML content in the given dictionary.

        Args:
            dct (dict): Dictionary containing HTML content.

        Returns:
            dict: Dictionary with the extracted and cleaned text.
        """
        text = (
            extract(
                dct[self.html_col],
                fast=True,
                output_format="markdown",
                target_language="zh",
                include_tables=True,
                include_comments=False,
                favor_precision=True,
                include_images=False,
                deduplicate=True,
            )
            or ""
        )
        if ftfy.is_bad(text):
            text = ftfy.fix_text(text)
            if ftfy.is_bad(text):
                text = ""
        dct[self.output_col] = text
        return dct

    def __call__(self, dcts: DictsGenerator) -> DictsGenerator:
        """
        Process a generator of dictionaries in parallel.

        Args:
            dcts (DictsGenerator): Generator of dictionaries to process.

        Yields:
            DictsGenerator: Generator of processed dictionaries.
        """
        with ProcessPoolExecutor(self.num_proc) as executor:
            yield from executor.map(self.run_each, dcts)
