from config_synth_flow.base import BasePipeline, DictsGenerator


class ListFlatter(BasePipeline):
    """
    A pipeline to flatten lists in a specified column of dictionaries.
    """

    def post_init(self, text_col: str):
        """
        Initialize the ListFlatter with the column name containing lists.

        Args:
            text_col (str): The name of the column containing lists to be flattened.
        """
        self.text_col = text_col

    def run_each(self, dct: dict) -> DictsGenerator:
        """
        Flatten the list in the specified column of the dictionary.

        Args:
            dct (dict): The dictionary containing the list to be flattened.

        Yields:
            dict: A dictionary with the flattened list elements.

        Raises:
            ValueError: If an unsupported type is encountered in the list.
        """
        assert isinstance(dct[self.text_col], list)
        text_list = dct.pop(self.text_col)
        for text in text_list:
            if isinstance(text, str):
                yield {**dct, self.text_col: text}
            elif isinstance(text, dict):
                yield {**dct, **text}
            else:
                raise ValueError(f"Unsupported type: {type(text)}")


class RemoveColumns(BasePipeline):
    """
    A pipeline to remove specified columns or hidden columns from dictionaries.
    """

    def post_init(self, remove_hidden_cols: bool = True, cols: list[str] = None):
        """
        Initialize the RemoveColumns with columns to remove and whether to remove hidden columns.

        Args:
            remove_hidden_cols (bool): Whether to remove columns starting with an underscore.
            cols (list[str], optional): A list of column names to remove. Defaults to None.
        """
        self.columns = cols or []
        self.remove_hidden_cols = remove_hidden_cols

    def run_each(self, dct: dict) -> dict:
        """
        Remove specified columns from the dictionary.

        Args:
            dct (dict): The dictionary from which columns will be removed.

        Returns:
            dict: The dictionary with specified columns removed.
        """
        keys: list[str] = list(dct.keys())
        for k in keys:
            if k in self.columns:
                dct.pop(k)
            elif self.remove_hidden_cols and k.startswith("_"):
                dct.pop(k)
        return dct
