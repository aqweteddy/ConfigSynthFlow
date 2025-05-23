from ..mixins import MultiProcessCoreMixin
from .base_pipeline import BasePipeline
from .config import MultiProcessConfig, PipelineConfig


class MultiProcessBasePipeline(BasePipeline, MultiProcessCoreMixin):
    """Pipeline for multi-process processing."""

    mp_cfg: MultiProcessConfig

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.mp_cfg = config.mp_cfg
