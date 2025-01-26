from ..base import BasePipeline, PipelineConfig
from ..reader import BaseReader
from ..writer import BaseWriter


class BaseExecutor(BasePipeline):
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.reader.set_writer_output_path(self.writer.output_path)
        
    def __post_init__(self, 
                      reader: BaseReader,
                      writer: BaseWriter,
    ):
        self.reader = reader
        self.writer = writer
