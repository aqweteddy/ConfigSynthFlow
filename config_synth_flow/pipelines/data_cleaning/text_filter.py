import re

from ...base import BasePipeline

CHINESE_CHAR = re.compile(r"[\u4e00-\u9fff]", flags=re.MULTILINE | re.IGNORECASE)
TOTAL_CHAR_EXCEPT_SPACE = re.compile(r"[^\s]", flags=re.MULTILINE | re.IGNORECASE)
SPLITERS = re.compile(r"[\n\r\t\s。？！；,，|]", flags=re.MULTILINE | re.IGNORECASE)


class TextFilter(BasePipeline):
    """Pipeline for filtering text based on various criteria."""

    def post_init(
        self,
        text_col: str = "text",
        min_char_length: int = 10,
        max_char_length: int = 100_000_00,
        min_chinese_char_ratio: float = 0,
        max_char_between_spliter: int = 200,
        stopwords: list[str] = None,
    ):
        """
        Initialize the pipeline with filtering criteria.

        Args:
            text_col (str): Column name containing text.
            min_char_length (int): Minimum character length of the text.
            max_char_length (int): Maximum character length of the text.
            min_chinese_char_ratio (float): Minimum ratio of Chinese characters in the text.
            max_char_between_spliter (int): Maximum characters allowed between splitters.
            stopwords (list[str]): List of stopwords to filter out.
        """
        self.min_char_length = min_char_length
        self.max_char_length = max_char_length
        self.min_chinese_char_ratio = min_chinese_char_ratio
        self.max_char_between_spliter = max_char_between_spliter
        self.stopwords = stopwords or [
            "javascript",
            "css",
        ]
        self.text_col = text_col

    def run_each(self, dct: dict) -> dict:
        """
        Filter text in the given dictionary based on the criteria.

        Args:
            dct (dict): Dictionary containing text.

        Returns:
            dict: Dictionary if text passes the filters, otherwise None.
        """
        text: str = dct[self.text_col]
        if len(text) < self.min_char_length or len(text) > self.max_char_length:
            return None
        if (
            self.min_chinese_char_ratio > 0
            and len(CHINESE_CHAR.findall(text))
            / len(TOTAL_CHAR_EXCEPT_SPACE.findall(text))
            < self.min_chinese_char_ratio
        ):
            return None
        for t in SPLITERS.split(text):
            if len(t) > self.max_char_between_spliter:
                return None
        for stopword in self.stopwords:
            if stopword in text.lower():
                return None
        return dct
