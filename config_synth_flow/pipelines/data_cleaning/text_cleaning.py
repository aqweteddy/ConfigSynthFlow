import re
from typing import Callable

from ...base import BasePipeline


def func_remove_reduent_spaces(text: str) -> str:
    """
    Remove redundant spaces and newlines from the text.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text.
    """
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s{3,}", "", text)
    text = re.sub("\\_", "_", text)
    return text


def func_shrink_md_table(text: str) -> str:
    """
    Shrink markdown table formatting.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text.
    """
    text = re.sub(r"-{3,}", "---", text)
    text = re.sub(r"\|{3,}", "|", text)
    return text


def get_custom_regex_sub(custom_regex_sub: list[str]) -> Callable:
    """
    Generate a function to apply custom regex substitutions.

    Args:
        custom_regex_sub (list[str]): List containing regex pattern and substitution.

    Returns:
        Callable: Function to apply the regex substitution.
    """

    def sub(text):
        return re.sub(custom_regex_sub[0], custom_regex_sub[1], text)

    return sub


def remove_line(
    text: str, min_char_length_per_line: int, max_char_length_per_line: int
) -> str:
    """
    Remove lines from the text based on character length criteria.

    Args:
        text (str): Input text.
        min_char_length_per_line (int): Minimum character length per line.
        max_char_length_per_line (int): Maximum character length per line.

    Returns:
        str: Cleaned text with lines removed.
    """
    result = []
    for line in text.split("\n"):
        if min_char_length_per_line > 0 and len(line) < min_char_length_per_line:
            continue
        if max_char_length_per_line > 0 and len(line) > max_char_length_per_line:
            continue
        result.append(line)
    return "\n".join(result)


class TextCleaning(BasePipeline):
    """Pipeline for cleaning text using various cleaning functions."""

    def post_init(
        self,
        text_col: str,
        min_char_length_per_line: int = -1,
        max_char_length_per_line: int = -1,
        remove_reduent_spaces: bool = False,
        shrink_md_table: bool = False,
        custom_regex_sub_list: list[list[str]] = None,
    ):
        """
        Initialize the pipeline with cleaning functions.

        Args:
            text_col (str): Column name containing text.
            min_char_length_per_line (int): Minimum character length per line.
            max_char_length_per_line (int): Maximum character length per line.
            remove_reduent_spaces (bool): Whether to remove redundant spaces.
            shrink_md_table (bool): Whether to shrink markdown table formatting.
            custom_regex_sub_list (list[list[str]]): List of custom regex substitutions.
        """
        self.clean_pipes = []
        self.text_col = text_col

        if remove_reduent_spaces:
            self.clean_pipes.append(func_remove_reduent_spaces)
        if shrink_md_table:
            self.clean_pipes.append(func_shrink_md_table)
        if custom_regex_sub_list:
            for custom_regex_sub in custom_regex_sub_list:
                self.clean_pipes.append(get_custom_regex_sub(custom_regex_sub))

        if min_char_length_per_line > 0 and max_char_length_per_line > 0:
            self.clean_pipes.append(
                lambda x: remove_line(
                    x, min_char_length_per_line, max_char_length_per_line
                )
            )

    def run_each(self, dct: dict) -> dict:
        """
        Clean text in the given dictionary using the cleaning functions.

        Args:
            dct (dict): Dictionary containing text.

        Returns:
            dict: Dictionary with cleaned text.
        """
        text = dct[self.text_col]
        for pipe in self.clean_pipes:
            text = pipe(text)
        dct[self.text_col] = text.strip()
        return dct
