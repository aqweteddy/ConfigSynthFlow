from ...base import BasePipeline
import re

CHINESE_CHAR = re.compile(r"[\u4e00-\u9fff]", flags=re.MULTILINE | re.IGNORECASE)
TOTAL_CHAR_EXCEPT_SPACE = re.compile(r"[^\s]", flags=re.MULTILINE | re.IGNORECASE)
SPLITERS = re.compile(r"[\n\r\t\s。？！；,，|]", flags=re.MULTILINE | re.IGNORECASE)


class TextFilter(BasePipeline):
    def __post_init__(
        self,
        text_col: str = "text",
        min_char_length: int = 10,
        max_char_length: int = 100_000_00,
        min_chinese_char_ratio: float = 0,
        max_char_between_spliter: int = 200,
        stopwords: list[str] = None,
    ):
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
