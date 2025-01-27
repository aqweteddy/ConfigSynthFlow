"""
This module contains the AccuracyMetric class, which is used to calculate the accuracy of predictions
compared to ground truth values, optionally grouped by a specified column.
"""

from collections import defaultdict

from ...base import BasePipeline, DictsGenerator


class AccuracyMetric(BasePipeline):
    """
    A class used to calculate accuracy metrics for predictions.

    Attributes:
        pred_lambda_col (str): A string representation of a lambda function for predictions.
        ground_lambda_col (str): A string representation of a lambda function for ground truth values.
        group_by_col (str): The column name to group results by.
        grp_dct (defaultdict): A dictionary to store grouped accuracy results.
        total_cnt (int): Total number of predictions.
        correct_cnt (int): Total number of correct predictions.
    """

    def post_init(
        self,
        pred_lambda_col: str = None,
        ground_lambda_col: str = None,
        group_by_col: str = None,
    ):
        """
        Initializes the AccuracyMetric with the given parameters.

        Args:
            pred_lambda_col (str): A string representation of a lambda function for predictions.
            ground_lambda_col (str): A string representation of a lambda function for ground truth values.
            group_by_col (str): The column name to group results by.
        """
        self.pred_lambda_col = eval(pred_lambda_col)
        self.ground_lambda_col = eval(ground_lambda_col)
        self.group_by_col = group_by_col

        self.grp_dct = defaultdict(list)
        self.total_cnt = 0
        self.correct_cnt = 0

    def __call__(self, dcts: DictsGenerator) -> DictsGenerator:
        """
        Processes a generator of dictionaries, calculating accuracy for each prediction.

        Args:
            dcts (DictsGenerator): A generator of dictionaries containing prediction data.

        Yields:
            DictsGenerator: The same generator of dictionaries after processing.
        """
        for dct in dcts:
            is_correct = self.pred_lambda_col(dct) == self.ground_lambda_col(dct)
            self.grp_dct[dct[self.group_by_col]].append(is_correct)
            self.total_cnt += 1
            self.correct_cnt += is_correct
            yield dct

    def __del__(self):
        """
        Logs the accuracy results when the object is deleted.
        """
        self.logger.info(f"Total count: {self.total_cnt}")
        self.logger.info(f"Correct count: {self.correct_cnt}")
        self.logger.info(f"Accuracy: {self.correct_cnt/self.total_cnt}")
        res_str = ""
        for k, v in self.grp_dct.items():
            total_cnt = len(v)
            correct_cnt = sum(v)
            res_str += f"Group {k} accuracy: {correct_cnt/total_cnt:.4f}\n"
        self.logger.info(res_str)
