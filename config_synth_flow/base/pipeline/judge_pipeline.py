from abc import abstractmethod
from typing import Any

from .async_pipeline import AsyncBasePipeline


class JudgePipeline(AsyncBasePipeline):
    """
    JudgePipeline is a pipeline that judges the quality of something in the Dict.
    It must define the `judge` method to judge the quality of the data and the `get_score` method to get the score of the `judge` result.
    """

    @abstractmethod
    async def judge(self, dct: dict) -> Any:
        """
        Judge the quality of the data.
        """
        ...

    @abstractmethod
    def get_score(self, judge_result: Any) -> float:
        """
        Get the score of the judge result.
        """
        ...
