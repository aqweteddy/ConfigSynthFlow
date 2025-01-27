"""
This module provides functionality for text chunking using a specified tokenizer and text splitter.

Classes:
    TextChunker: A pipeline component for chunking text into smaller pieces.
"""

from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer

from ...base import BasePipeline


class TextChunker(BasePipeline):
    def post_init(
        self,
        text_lambda_col: str,
        tokenizer_path: str,
        output_col: str = "_chunked",
        chunk_size: tuple[int, int] = (300, 1000),
        overlap: int = 50,
    ):
        """
        Initializes the TextChunker with the given parameters.

        Args:
            text_lambda_col (str): A string representation of a lambda function to extract text from input dictionary.
            tokenizer_path (str): Path to the tokenizer file or pretrained tokenizer name.
            output_col (str, optional): The key for the output chunks in the resulting dictionary. Defaults to '_chunked'.
            chunk_size (tuple[int, int], optional): A tuple specifying the minimum and maximum chunk sizes. Defaults to (300, 1000).
            overlap (int, optional): The number of overlapping tokens between chunks. Defaults to 50.
        """
        self.text_lambda_col = eval(text_lambda_col)
        try:
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
        except Exception as e:
            self.logger.info(
                f"Failed to load tokenizer from file: {e}, trying to load from hugginface."
            )
            self.tokenizer = Tokenizer.from_pretrained(tokenizer_path)

        self.splitter = TextSplitter.from_huggingface_tokenizer(
            self.tokenizer, (chunk_size[0], chunk_size[1]), overlap=overlap
        )

        self.output_col = output_col

    def run_each(self, dct: dict) -> dict:
        """
        Processes a single dictionary to chunk the text.

        Args:
            dct (dict): The input dictionary containing text to be chunked.

        Returns:
            dict: The input dictionary with an additional key for the chunked text.
        """
        text = self.text_lambda_col(dct)
        if isinstance(text, list):
            result = []
            for t in text:
                chunks = self.splitter.chunks(t)
                result.extend(chunks)
        else:
            result = self.splitter.chunks(text)
        return {**dct, self.output_col: result}
