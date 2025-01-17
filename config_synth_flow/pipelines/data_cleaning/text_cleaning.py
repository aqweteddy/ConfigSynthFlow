from ...base import BasePipeline
import re
from typing import Callable


def func_remove_reduent_spaces(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s{3,}", "", text)
    text = re.sub("\\_", "_", text)
    return text


def func_shrink_md_table(text: str) -> str:
    text = re.sub(r"-{3,}", "---", text)
    text = re.sub(r"\|{3,}", "|", text)
    return text


def get_custom_regex_sub(custom_regex_sub: list[str]) -> Callable:
    def sub(text):
        return re.sub(custom_regex_sub[0], custom_regex_sub[1], text)

    return sub


class TextCleaning(BasePipeline):
    def __post_init__(
        self,
        text_col: str,
        remove_reduent_spaces: bool = False,
        shrink_md_table: bool = False,
        custom_regex_sub_list: list[list[str]] = None,
    ):
        self.clean_pipes = []
        self.text_col = text_col
        if remove_reduent_spaces:
            self.clean_pipes.append(func_remove_reduent_spaces)
        if shrink_md_table:
            self.clean_pipes.append(func_shrink_md_table)
        if custom_regex_sub_list:
            for custom_regex_sub in custom_regex_sub_list:
                self.clean_pipes.append(get_custom_regex_sub(custom_regex_sub))

    def run_each(self, dct: dict) -> dict:
        text = dct[self.text_col]
        for pipe in self.clean_pipes:
            text = pipe(text)
        dct[self.text_col] = text.strip()
        return dct
