"""
Mixin class for core pipeline functionality.
"""

import asyncio
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from typing import Any

from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio as atqdm

from ..pipeline.config import AsyncConfig, MultiProcessConfig
from .types import DictsGenerator


class PipelineCoreMixin:
    """Mixin for core pipeline functionality"""

    mp_cfg: MultiProcessConfig

    def post_init(self, **kwargs: Any) -> None:
        """
        Post initialization method. Write Pipeline initialization code here.

        Args:
            kwargs (Any): Initial kwargs.
        """
        ...

    @property
    def class_name(self) -> str:
        """
        Get the class name of the pipeline.
        """
        return self.__class__.__name__

    def run_each(self, dct: dict) -> dict | DictsGenerator | None:
        """
        Run the pipeline on each input sample.

        Args:
            dct (dict): Input dictionary.
        Returns:
            dict | Sequence[dict] | None: Output dictionary.
        """
        ...

    def __call__(self, dcts: DictsGenerator) -> DictsGenerator:
        """
        Run the pipeline on a generator of input samples.

        Args:
            dcts (DictsGenerator): Input samples.
        Returns:
            DictsGenerator: Generator of output samples.
        """

        for dct in dcts:
            result = self.run_each(dct)
            if result:
                if isinstance(result, dict):
                    yield result
                else:
                    yield from result


class MultiProcessCoreMixin(PipelineCoreMixin):
    """Mixin for core pipeline functionality"""

    mp_cfg: MultiProcessConfig

    def __call__(self, dcts: DictsGenerator) -> DictsGenerator:
        """
        Run the pipeline on a generator of input samples.
        """
        if self.mp_cfg.num_proc is None or self.mp_cfg.num_proc == 1:
            yield from super().__call__(dcts)
        else:
            yield from self._multi_process_call(dcts)

    def _multi_process_call(self, dcts: DictsGenerator) -> DictsGenerator:
        """
        Run the pipeline on a generator of input samples in multi-process mode.

        Args:
            dcts (DictsGenerator): Input samples.
        Returns:
            DictsGenerator: Generator of output samples.
        """
        # Create num_proc object to execute dcts in parallel.
        with ProcessPoolExecutor(max_workers=self.mp_cfg.num_proc) as executor:
            # Map run_each to each dictionary in the list
            results = executor.map(self.run_each, dcts)

            # Process and yield the results
            for result in tqdm(results, total=len(dcts), desc=self.class_name):
                if result:
                    if isinstance(result, dict):
                        yield result
                    else:
                        yield from result


class AsyncCoreMixin(PipelineCoreMixin):
    """Mixin for core pipeline functionality"""

    async_cfg: AsyncConfig

    @property
    def semaphore(self):
        """
        Get the semaphore for limiting concurrency.

        Returns:
            asyncio.Semaphore: Semaphore for limiting concurrency.
        """
        return asyncio.Semaphore(self.config.async_cfg.concurrency)

    async def run_each(self, dct: dict) -> dict | DictsGenerator | None:
        """
        Run the pipeline on each input sample asynchronously.

        Args:
            dct (dict): Input dictionary.
        Returns:
            dict | Sequence[dict] | None: Output dictionary.
        """
        raise NotImplementedError(f"{self.class_name} is missing `run_each` method")

    async def _async_call(self, dcts: list[dict]) -> list[dict]:
        """
        Run the pipeline on a list of input samples asynchronously.

        Args:
            dcts (list[dict]): List of input samples.

        Returns:
            list[dict]: List of output samples.
        """
        sem = asyncio.Semaphore(self.config.async_cfg.concurrency)

        async def limit_concurrency(sem, coro):
            async with sem:
                return await coro

        tasks = []
        for dct in dcts:
            tasks.append(limit_concurrency(sem, self.run_each(dct)))

        return await atqdm.gather(
            *tasks,
            desc=self.class_name,
            disable=not self.config.async_cfg.show_progress,
        )

    def __get_async_chunks(self, dcts: DictsGenerator, chunk_size: int) -> DictsGenerator:
        """
        Get chunks of input samples.

        Args:
            dcts (DictsGenerator): Input samples.
            chunk_size (int): Size of each chunk.

        Returns:
            DictsGenerator: Generator of chunks of input samples.
        """
        it = iter(dcts)
        while chunk := list(islice(it, chunk_size)):
            yield list(chunk)

    def __call__(self, dcts: DictsGenerator) -> DictsGenerator:
        """
        Yield the output of the pipeline in chunks.

        Args:
            dcts (DictsGenerator): Input samples.
        Returns:
            DictsGenerator: Generator of output samples.
        """
        for chunk in self.__get_async_chunks(dcts, self.config.async_cfg.batch_size):
            for dct in asyncio.run(self._async_call(chunk)):
                if isinstance(dct, dict):
                    yield dct
                elif isinstance(dct, list):
                    yield from dct
