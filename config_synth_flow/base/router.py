from . import BasePipeline

class BaseRounter(BasePipeline):
    def __post_init__(self, **kwargs):
        return super().__post_init__(**kwargs)