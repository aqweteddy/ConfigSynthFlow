import re

from ...base import BasePipeline


class MarkdownCleaning(BasePipeline):
    """Pipeline for cleaning text using various cleaning functions."""

    def post_init(
        self,
        text_col: str,
        remove_url: bool = False,
        remove_image: bool = False,
        remove_reduent_spaces: bool = False,
    ):
        """
        Initialize the pipeline with cleaning options.

        Args:
            text_col (str): Name of the text column to clean
            remove_url (bool): Whether to remove URLs
            remove_image (bool): Whether to remove image markdown
            remove_reduent_spaces (bool): Whether to remove redundant spaces
        """
        self.text_col = text_col
        self.cleaning_functions = []

        if remove_url:
            self.cleaning_functions.append(self._remove_urls)
        if remove_image:
            self.cleaning_functions.append(self._remove_images)
        if remove_reduent_spaces:
            self.cleaning_functions.append(self._remove_redundant_spaces)

    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        return re.sub(url_pattern, "", text)

    def _remove_images(self, text: str) -> str:
        """Remove markdown image syntax."""
        image_pattern = r"!\[.*?\]\(.*?\)"
        return re.sub(image_pattern, "", text)

    def _remove_redundant_spaces(self, text: str) -> str:
        """Remove redundant spaces and newlines."""
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"\s{3,}", "", text)
        return text.strip()

    def run_each(self, dct: dict) -> dict:
        """
        Clean text in the given dictionary using the cleaning functions.

        Args:
            dct (dict): Dictionary containing text.

        Returns:
            dict: Dictionary with cleaned text.
        """
        text = dct[self.text_col]

        for clean_func in self.cleaning_functions:
            text = clean_func(text)

        dct[self.text_col] = text
        return dct
