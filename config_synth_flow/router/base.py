import random

from ..base import AsyncBasePipeline, BasePipeline


class BaseRouter(BasePipeline):
    def post_init(
        self,
        pipeline_list: list[dict],
    ):
        """
        Initialize the BaseRouter.
        Notice that the pipeline in the `pipeline_list` should implement `run_each` method.
        Args:
            pipeline_list (list[dict]): List of dictionaries containing pipeline configurations.
        """
        self.pipeline_list = pipeline_list

    def get_random_pipeline(self) -> BasePipeline:
        """
        Get a random pipeline from the pipeline list.
        Returns:
            BasePipeline: Random pipeline from the pipeline list.
        """
        return random.choices(
            self.pipeline_list, weights=[item.weight for item in self.pipeline_list]
        )[0].pipeline


class AsyncBaseRounter(AsyncBasePipeline, BaseRouter):
    def post_init(
        self,
        pipeline_list: list[dict],
    ):
        """
        Initialize the BaseRouter.
        Notice that the pipeline in the `pipeline_list` should implement `run_each` method.
        Args:
            pipeline_list (list[dict]): List of dictionaries containing pipeline configurations.
        """
        self.pipeline_list = pipeline_list
