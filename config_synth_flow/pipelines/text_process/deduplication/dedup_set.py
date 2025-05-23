"""
This module provides a deduplication pipeline using exact text matching.
"""

from config_synth_flow.base import BasePipeline, DictsGenerator


class SetExactMatchDeduplicator(BasePipeline):
    """
    A pipeline for deduplicating documents based on exact text matches.

    Attributes:
        text_col (str): The name of the text column to be used for deduplication.
    """

    def post_init(self, text_col: str = "text"):
        """
        Initializes the pipeline with the specified text column.

        Args:
            text_col (str): The name of the text column to be used for deduplication.
        """
        self.text_col = text_col

    def __call__(self, dcts: DictsGenerator):
        """
        Deduplicates the input documents based on exact text matches.

        Args:
            dcts (DictsGenerator): A generator of dictionaries representing documents.

        Yields:
            dict: A dictionary representing a unique document.
        """
        used = set()

        for dct in dcts:
            text = dct[self.text_col]
            if text not in used:
                yield dct
                used.add(text)
