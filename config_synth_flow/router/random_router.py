import asyncio

from .base import AsyncBaseRounter, BaseRouter


class RandomRouter(BaseRouter):
    def run_each(self, dct: dict) -> dict:
        """
        Run the pipeline on each dictionary.
        Args:
            dct (dict): Dictionary to run the pipeline on.
        Returns:
            dict: Dictionary after running the pipeline.
        """
        pipeline = self.get_random_pipeline()
        return pipeline.run_each(dct)


class AsyncRandomRouter(AsyncBaseRounter):
    async def run_each(self, dct: dict) -> dict:
        """
        Run the pipeline on each dictionary asynchronously.
        Args:
            dct (dict): Dictionary to run the pipeline on.
        Returns:
            dict: Dictionary after running the pipeline.
        """
        pipeline = self.get_random_pipeline()
        if asyncio.iscoroutinefunction(pipeline.run_each):
            return await pipeline.run_each(dct)
        return pipeline.run_each(dct)
