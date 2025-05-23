"""
Asynchronous pipeline implementation.
"""

from ..mixins import AsyncCoreMixin
from .base_pipeline import BasePipeline
from .config import PipelineConfig


class AsyncBasePipeline(BasePipeline, AsyncCoreMixin):
    """Pipeline for asynchronous processing."""

    def __init__(self, config: PipelineConfig):
        """
        Initialize the asynchronous pipeline.

        Args:
            config (PipelineConfig): Pipeline configuration.
        """
        super().__init__(config)
        config.async_cfg.batch_size = config.async_cfg.batch_size or 1000
        config.async_cfg.concurrency = config.async_cfg.concurrency or 100
